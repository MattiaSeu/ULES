import copy
import math
import warnings
from functools import partial
from typing import Optional
from typing import Union

import ipdb

import attr
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms.functional
from pytorch_lightning.utilities import AttributeDict
from torch.utils.data import DataLoader
from torchvision import transforms

import self_sup_utils
from batchrenorm import BatchRenorm1d
from lars import LARS
from model_params import ModelParams
from sklearn.linear_model import LogisticRegression
import numpy as np

import os
from data_loading import CityData, CityDataContrastive
from utils.tenprint import print_tensor
from collections import OrderedDict
from torch.nn.functional import normalize


def get_mlp_normalization(hparams: ModelParams, prediction=False):
    normalization_str = hparams.mlp_normalization
    if prediction and hparams.prediction_mlp_normalization != "same":
        normalization_str = hparams.prediction_mlp_normalization

    if normalization_str is None:
        return None
    elif normalization_str == "bn":
        return partial(torch.nn.BatchNorm1d, num_features=hparams.mlp_hidden_dim)
    elif normalization_str == "br":
        return partial(BatchRenorm1d, num_features=hparams.mlp_hidden_dim)
    elif normalization_str == "ln":
        return partial(torch.nn.LayerNorm, normalized_shape=[hparams.mlp_hidden_dim])
    elif normalization_str == "gn":
        return partial(torch.nn.GroupNorm, num_channels=hparams.mlp_hidden_dim, num_groups=32)
    else:
        raise NotImplementedError(f"mlp normalization {normalization_str} not implemented")


def collate_data(data):
    img, lbl = list(zip(*data))
    lbl = [transforms.Resize((224, 224))(i) for i in lbl]
    lbl = [transforms.functional.pil_to_tensor(i) for i in lbl]
    img = torch.stack(img)
    lbl = torch.vstack(lbl)
    return img, lbl


class SelfSupervisedMethod(pl.LightningModule):
    model: torch.nn.Module
    dataset: self_sup_utils.DatasetBase
    hparams: AttributeDict
    embedding_dim: Optional[int]

    def __init__(
            self,
            hparams: Union[ModelParams, dict, None] = None,
            **kwargs,
    ):
        super().__init__()

        if hparams is None:
            hparams = self.params(**kwargs)
        elif isinstance(hparams, dict):
            hparams = self.params(**hparams, **kwargs)

        if isinstance(self.hparams, AttributeDict):
            self.hparams.update(AttributeDict(attr.asdict(hparams)))
        else:
            self.hparams = AttributeDict(attr.asdict(hparams))

        # Check for configuration issues
        if (
                hparams.gather_keys_for_queue
                and not hparams.shuffle_batch_norm
                and not hparams.encoder_arch.startswith("ws_")
        ):
            warnings.warn(
                "Configuration suspicious: gather_keys_for_queue without shuffle_batch_norm or weight standardization"
            )

        some_negative_examples = hparams.use_negative_examples_from_batch or hparams.use_negative_examples_from_queue
        if hparams.loss_type == "ce" and not some_negative_examples:
            warnings.warn("Configuration suspicious: cross entropy loss without negative examples")

        # Create encoder model
        self.model = self_sup_utils.get_encoder(hparams.encoder_arch, hparams.dataset_name,
                                                hparams.batch_size, num_classes=20)

        # Create dataset
        if hparams.encoder_arch != 'FCN_resnet50':
            self.dataset = self_sup_utils.get_moco_dataset(hparams)
            self.dataset_name = None
        else:
            self.dataset_name = hparams.dataset_name
            city_data_path = os.path.join("~/data", 'cityscapes_segments/')
            if hparams.extra:
                self.train_class = CityDataContrastive(city_data_path, split='train_extra', mode='coarse',
                                                       target_type='color', transforms=None)
            else:
                self.train_class = CityDataContrastive(city_data_path, split='train', mode='fine',
                                                       target_type='semantic', transforms=None)
            self.val_class = CityDataContrastive(city_data_path, split='val', mode='coarse',
                                                 target_type='color', transforms=None)

        if hparams.use_lagging_model:
            # "key" function (no grad)
            self.lagging_model = copy.deepcopy(self.model)
            for param in self.lagging_model.parameters():
                param.requires_grad = False
        else:
            self.lagging_model = None

        # self.projection_model = self_sup_utils.MLP(
        #     hparams.embedding_dim,
        #     hparams.dim,
        #     hparams.mlp_hidden_dim,
        #     num_layers=hparams.projection_mlp_layers,
        #     normalization=get_mlp_normalization(hparams),
        #     weight_standardization=hparams.use_mlp_weight_standardization,
        # )

        self.projection_model = torch.nn.Identity()

        self.prediction_model = self_sup_utils.MLP(
            hparams.dim,
            hparams.dim,
            hparams.mlp_hidden_dim,
            num_layers=hparams.prediction_mlp_layers,
            normalization=get_mlp_normalization(hparams, prediction=True),
            weight_standardization=hparams.use_mlp_weight_standardization,
        )

        # self.prediction_model = torch.nn.Identity()

        if hparams.use_lagging_model:
            #  "key" function (no grad)
            self.lagging_projection_model = copy.deepcopy(self.projection_model)
            for param in self.lagging_projection_model.parameters():
                param.requires_grad = False
        else:
            self.lagging_projection_model = None

        # this classifier is used to compute representation quality each epoch
        self.sklearn_classifier = LogisticRegression(max_iter=1000, solver="liblinear")

        if hparams.use_negative_examples_from_queue:
            # create the queue
            self.register_buffer("queue", torch.randn(hparams.dim, hparams.K))
            self.queue = torch.nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        else:
            self.queue = None

    def _get_embeddings(self, x):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if isinstance(x, list):
            im_q = x[0].contiguous()
            im_k = x[1].contiguous()
        else:
            if len(x.shape) == 5:
                bsz, nd, nc, nh, nw = x.shape
                assert nd == 2, "second dimension should be the split image -- dims should be N2CHW"
                im_q = x[:, 0].contiguous()
                im_k = x[:, 1].contiguous()
            elif len(x.shape) == 4:
                bsz, nc, nh, nw = x.shape
                im_q = x[0].contiguous()
                im_k = x[1].contiguous()

        # compute query features

        if self.hparams.encoder_arch == 'FCN_resnet50':
            emb_q = self.model(im_q)
        else:
            emb_q = self.model(im_q)

        q_projection = self.projection_model(
            emb_q['out'].squeeze(0) if isinstance(emb_q, OrderedDict) else emb_q[..., 0, 0])
        q = self.prediction_model(q_projection)  # queries: NxC
        if self.hparams.use_lagging_model:
            # compute key features
            with torch.no_grad():  # no gradient to keys
                if self.hparams.shuffle_batch_norm:
                    im_k, idx_unshuffle = self_sup_utils.BatchShuffleDDP.shuffle(im_k)
                k = self.lagging_projection_model(self.lagging_model(im_k))  # keys: NxC
                if self.hparams.shuffle_batch_norm:
                    k = self_sup_utils.BatchShuffleDDP.unshuffle(k, idx_unshuffle)
        else:
            if self.hparams.encoder_arch == 'FCN_resnet50':
                emb_k = self.model(im_k)
            else:
                emb_k = self.model(im_k)

            k_projection = self.projection_model(
                emb_k['out'].squeeze(0) if isinstance(emb_k, OrderedDict) else emb_k[..., 0, 0])
            k = self.prediction_model(k_projection)  # queries: NxC

        if self.hparams.use_unit_sphere_projection:
            q = torch.nn.functional.normalize(q, dim=1)
            k = torch.nn.functional.normalize(k, dim=1)

        return emb_q, emb_k, q, k

    def _get_contrastive_predictions(self, q, k):
        if self.hparams.use_negative_examples_from_batch:
            logits = torch.mm(q, k.T)
            labels = torch.arange(0, q.shape[0], dtype=torch.long).to(logits.device)
            return logits, labels

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        q = q.permute(0, 2, 3, 1)
        q = q.reshape(-1, 20)
        k = k.permute(0, 2, 3, 1)
        k = k.reshape(-1, 20)
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        if self.hparams.use_negative_examples_from_queue:
            # negative logits: NxK
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
            logits = torch.cat([l_pos, l_neg], dim=1)
        else:
            logits = l_pos

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        return logits, labels

    def _get_pos_neg_ip(self, emb_q, k):
        with torch.no_grad():
            z = self.projection_model(emb_q['out'] if isinstance(emb_q, OrderedDict) else emb_q[..., 0, 0])
            z = torch.nn.functional.normalize(z, dim=1)
            ip = torch.mm(z, k.T)
            # ip = torch.matmul(z, k.T)
            eye = torch.eye(z.shape[0]).to(z.device)
            pos_ip = (ip * eye).sum() / z.shape[0]
            neg_ip = (ip * (1 - eye)).sum() / (z.shape[0] * (z.shape[0] - 1))

        return pos_ip, neg_ip

    def _get_contrastive_loss(self, logits, labels):
        if self.hparams.loss_type == "ce":
            if self.hparams.use_eqco_margin:
                if self.hparams.use_negative_examples_from_batch:
                    neg_factor = self.hparams.eqco_alpha / self.hparams.batch_size
                elif self.hparams.use_negative_examples_from_queue:
                    neg_factor = self.hparams.eqco_alpha / self.hparams.K
                else:
                    raise Exception("Must have negative examples for ce loss")

                predictions = self_sup_utils.log_softmax_with_factors(logits / self.hparams.T, neg_factor=neg_factor)
                return F.nll_loss(predictions, labels)

            return F.cross_entropy(logits / self.hparams.T, labels)

        new_labels = torch.zeros_like(logits)
        new_labels.scatter_(1, labels.unsqueeze(1), 1)
        if self.hparams.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits / self.hparams.T, new_labels) * logits.shape[1]

        if self.hparams.loss_type == "ip":
            # inner product
            # negative sign for label=1 (maximize ip), positive sign for label=0 (minimize ip)
            inner_product = (1 - new_labels * 2) * logits
            return torch.mean((inner_product + 1).sum(dim=-1))

        raise NotImplementedError(f"Loss function {self.hparams.loss_type} not implemented")

    def _get_vicreg_loss(self, z_a, z_b, batch_idx):
        z_a = z_a.permute(0, 2, 3, 1)
        z_a = z_a.reshape(-1, 20)
        z_b = z_b.permute(0, 2, 3, 1)
        z_b = z_b.reshape(-1, 20)
        assert z_a.shape == z_b.shape and len(z_a.shape) == 2

        # invariance loss
        loss_inv = F.mse_loss(z_a, z_b)

        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.hparams.variance_loss_epsilon)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.hparams.variance_loss_epsilon)
        loss_v_a = torch.mean(F.relu(1 - std_z_a))
        loss_v_b = torch.mean(F.relu(1 - std_z_b))
        loss_var = loss_v_a + loss_v_b

        # covariance loss
        N, D = z_a.shape
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = ((z_a.T @ z_a) / (N - 1)).square()  # DxD
        cov_z_b = ((z_b.T @ z_b) / (N - 1)).square()  # DxD
        loss_c_a = (cov_z_a.sum() - cov_z_a.diagonal().sum()) / D
        loss_c_b = (cov_z_b.sum() - cov_z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b

        weighted_inv = loss_inv * self.hparams.invariance_loss_weight
        weighted_var = loss_var * self.hparams.variance_loss_weight
        weighted_cov = loss_cov * self.hparams.covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov

        return {
            "loss": loss,
            "loss_invariance": weighted_inv,
            "loss_variance": weighted_var,
            "loss_covariance": weighted_cov,
        }

    def _get_custom_loss(self, q, k, superpix_x, superpix_t, flip_flag, crop_pos):

        # initialize the losses
        pos_loss = 0.0
        neg_loss = 0.0
        var_loss = 0.0
        cos_sim = torch.nn.CosineSimilarity(dim=0)

        onehot_masks, mask_values_list = self.segment_to_onehot(superpix_x)

        # iterate through the batch
        for batch_idx in range(q.shape[0]):
            mask_values = mask_values_list[batch_idx]  # store mask values for current image
            mask_n = len(onehot_masks[batch_idx])  # store the number of masks for the current image
            if flip_flag[batch_idx]:
                q_i = torchvision.transforms.functional.hflip(q[batch_idx])
            else:
                q_i = q[batch_idx]
            k_i = k[batch_idx]

            q_i = torch.nn.functional.normalize(q_i)
            k_i = torch.nn.functional.normalize(k_i)
            for mask_idx in range(mask_n):  # iterate through the mask
                mask = onehot_masks[batch_idx][mask_idx]
                mask_k = (superpix_t[batch_idx] == mask_values[mask_idx])
                if mask_k.sum() != 0:  # check if the mask actually has correspondence

                    pos_loss += torch.abs(((q_i * mask).mean((1, 2)))
                                          - (k_i * mask_k).mean((1, 2))).sum()

                mean_qi = (q_i * mask).mean((1, 2))
                mean_neg_qi = (q_i * torch.logical_not(mask)).mean((1, 2))

                # normalize between 0 and 1
                mean_qi -= mean_qi.clone().min()
                mean_qi += 1e-12
                mean_qi /= mean_qi.clone().max()

                mean_neg_qi -= mean_neg_qi.clone().min()
                mean_neg_qi += 1e-12
                mean_neg_qi /= mean_neg_qi.clone().max()

                neg_loss += cos_sim(mean_qi, mean_neg_qi)
                var_loss += (q_i * mask).std()

        loss = pos_loss + neg_loss + var_loss
        return {
            "loss": loss,
            "pos_loss": pos_loss,
            "neg_loss": neg_loss,
            "var_loss": var_loss,
        }

    def _get_custom_contrastive_loss(self, q, k, superpix_x, superpix_t, flip_flag, crop_pos):

        # initialize the losses
        loss = 0.0
        cos_sim = torch.nn.CosineSimilarity(dim=0)

        onehot_masks, mask_values_list = self.segment_to_onehot(superpix_x)

        # iterate through the batch
        for batch_idx in range(q.shape[0]):
            mask_values = mask_values_list[batch_idx]  # store mask values for current image
            mask_n = len(onehot_masks[batch_idx])  # store the number of masks for the current image
            if flip_flag[batch_idx]:
                q_i = torchvision.transforms.functional.hflip(q[batch_idx])
            else:
                q_i = q[batch_idx]
            k_i = k[batch_idx]

            # q_i = torch.nn.functional.normalize(q_i)
            # k_i = torch.nn.functional.normalize(k_i)
            sim_qk = torch.empty([1])
            for mask_idx in range(mask_n):  # iterate through the mask
                mask = onehot_masks[batch_idx][mask_idx]
                mask_k = (superpix_t[batch_idx] == mask_values[mask_idx])
                if mask_k.sum() != 0:  # check if the mask actually has correspondence
                    mean_qi = (q_i * mask).mean((1, 2))
                    mean_ki = (k_i * mask_k).mean((1, 2))
                    mean_qi = torch.nn.functional.normalize(mean_qi, dim=-1)
                    mean_ki = torch.nn.functional.normalize(mean_ki, dim=-1)
                    sim_qk = (mean_qi @ mean_ki.T) / 0.1

                loss += torch.nn.functional.cross_entropy(sim_qk, 20)

        return {
            "loss": loss,
        }

    def segment_to_onehot(self, superpix_seg):
        one_hot_masks = []
        mask_values_list = []
        for superpix_seg_single in superpix_seg:
            mask_values = superpix_seg_single.unique().tolist()
            mask_list = []

            for label in mask_values:
                mask = torch.zeros_like(superpix_seg_single)
                mask[superpix_seg_single == label] = 1
                mask_list.append(mask)

            one_hot_masks.append(mask_list)
            mask_values_list.append(mask_values)

        return one_hot_masks, mask_values_list

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        x = [batch['image'], batch['img_contrastive']]
        superpix_x = batch['target']
        superpix_t = batch['target_t']
        flip_flag = batch['flip_flag']
        crop_pos = batch['crop_pos']  # batch is a tuple, we just want the image

        emb_q, emb_k, q, k = self._get_embeddings(x)

        logits, labels = self._get_contrastive_predictions(q, k)
        if self.hparams.use_vicreg_loss:
            # losses = self._get_vicreg_loss(q, k, batch_idx)
            # losses = self._get_custom_loss(q, k, superpix_x, superpix_t, flip_flag, crop_pos)
            losses = self._get_custom_contrastive_loss(q, k, superpix_x, superpix_t, flip_flag, crop_pos)
            contrastive_loss = losses["loss"]
        else:
            losses = {}
            contrastive_loss = self._get_contrastive_loss(logits, labels)

            if self.hparams.use_both_augmentations_as_queries:
                x_flip = torch.flip(x, dims=[1])
                emb_q2, q2, k2 = self._get_embeddings(x_flip)
                logits2, labels2 = self._get_contrastive_predictions(q2, k2)

                pos_ip2, neg_ip2 = self._get_pos_neg_ip(emb_q2, k2)
                pos_ip = (pos_ip + pos_ip2) / 2
                neg_ip = (neg_ip + neg_ip2) / 2
                contrastive_loss += self._get_contrastive_loss(logits2, labels2)

        contrastive_loss = contrastive_loss.mean() * self.hparams.loss_constant_factor

        log_data = {
            "step_train_loss": contrastive_loss,
            "step_pos_cos": 0,
            "step_neg_cos": 0,
            **losses,
        }

        with torch.no_grad():
            self._momentum_update_key_encoder()

        some_negative_examples = (
                self.hparams.use_negative_examples_from_batch or self.hparams.use_negative_examples_from_queue
        )
        if some_negative_examples:
            acc1, acc5 = self_sup_utils.calculate_accuracy(logits, labels, topk=(1, 5))
            log_data.update({"step_train_acc1": acc1, "step_train_acc5": acc5})

        # dequeue and enqueue
        if self.hparams.use_negative_examples_from_queue:
            self._dequeue_and_enqueue(k)

        self.log_dict(log_data)
        return {"loss": contrastive_loss}

    def validation_step(self, batch, batch_idx):
        pass
        x = [batch['image'], batch['img_contrastive']]
        superpix_x = batch['target']
        flip_flag = batch['flip_flag']
        crop_pos = batch['crop_pos']  # batch is a tuple, we just want the image

        with torch.no_grad():
            # emb = self.model(x['out'] if isinstance(x, OrderedDict) else x)
            emb = self.model(x[0])
        return {"emb": emb, 'superpix': superpix_x}

    def validation_epoch_end(self, outputs):
        print('validation time (?)')

    def configure_optimizers(self):
        # exclude bias and batch norm from LARS and weight decay
        regular_parameters = []
        regular_parameter_names = []
        excluded_parameters = []
        excluded_parameter_names = []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad is False:
                continue
            if any(x in name for x in self.hparams.exclude_matching_parameters_from_lars):
                excluded_parameters.append(parameter)
                excluded_parameter_names.append(name)
            else:
                regular_parameters.append(parameter)
                regular_parameter_names.append(name)

        param_groups = [
            {"params": regular_parameters, "names": regular_parameter_names, "use_lars": True},
            {
                "params": excluded_parameters,
                "names": excluded_parameter_names,
                "use_lars": False,
                "weight_decay": 0,
            },
        ]
        if self.hparams.optimizer_name == "sgd":
            optimizer = torch.optim.SGD
        elif self.hparams.optimizer_name == "lars":
            optimizer = partial(LARS, warmup_epochs=self.hparams.lars_warmup_epochs, eta=self.hparams.lars_eta)
        elif self.hparams.optimizer_name == "adam":
            optimizer = torch.optim.AdamW(param_groups)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.hparams.max_epochs,
                eta_min=self.hparams.final_lr_schedule_value,
            )
            return [optimizer], [self.lr_scheduler]
        else:
            raise NotImplementedError(f"No such optimizer {self.hparams.optimizer_name}")

        encoding_optimizer = optimizer(
            param_groups,
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            encoding_optimizer,
            self.hparams.max_epochs,
            eta_min=self.hparams.final_lr_schedule_value,
        )
        return [encoding_optimizer], [self.lr_scheduler]

    def _get_m(self):
        if self.hparams.use_momentum_schedule is False:
            return self.hparams.m
        return 1 - (1 - self.hparams.m) * (math.cos(math.pi * self.current_epoch / self.hparams.max_epochs) + 1) / 2

    def _get_temp(self):
        return self.hparams.T

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        if not self.hparams.use_lagging_model:
            return
        m = self._get_m()
        for param_q, param_k in zip(self.model.parameters(), self.lagging_model.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)
        for param_q, param_k in zip(self.projection_model.parameters(), self.lagging_projection_model.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.hparams.gather_keys_for_queue:
            keys = self_sup_utils.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.K  # move pointer

        self.queue_ptr[0] = ptr

    def train_dataloader(self):
        if self.dataset_name == "cityscapes":
            return DataLoader(self.train_class, batch_size=self.hparams.batch_size,
                              shuffle=True, num_workers=self.hparams.num_data_workers, pin_memory=True)
        else:
            return DataLoader(
                self.dataset.get_train(),
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_data_workers,
                pin_memory=self.hparams.pin_data_memory,
                drop_last=self.hparams.drop_last_batch,
                shuffle=True,
            )

    def val_dataloader(self):
        if self.dataset_name == "cityscapes":
            return DataLoader(self.val_class, batch_size=self.hparams.batch_size,
                              shuffle=False, num_workers=self.hparams.num_data_workers, pin_memory=True)
        else:
            return DataLoader(
                self.dataset.get_validation(),
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_data_workers,
                pin_memory=self.hparams.pin_data_memory,
                drop_last=self.hparams.drop_last_batch,
            )

    @classmethod
    def params(cls, **kwargs) -> ModelParams:
        return ModelParams(**kwargs)
