'''
Unsupervised Learning Enhanced Segmentation
'''
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import pytorch_lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from pytorch_lightning import LightningModule
from models.blocks import DownsamplerBlock, UpsamplerBlock, non_bottleneck_1d, SqueezeAndExciteFusionAdd, \
    AdaptivePyramidPoolingModule
from models.loss import mIoULoss, BinaryFocalLoss, CrossEntropyLoss, AsymmetricUnifiedFocalLoss
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt
from utils.segmap import encode_segmap, decode_segmap, kitti_encode, kitti_decode, multi_mat_decode, visnir_decode
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss, JaccardLoss, FocalLoss, TverskyLoss
from models.DINOv2 import Dinov2ForSemanticSegmentation
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling


class DbSegmenter(pytorch_lightning.LightningModule):
    def __init__(self, backbone_rgb, backbone_range, head, input_size: Optional[List[int]],
                 fusion_method='add', esa_arch=False, context=False):
        super().__init__()
        self.backbone_rgb = backbone_rgb
        self.backbone_range = backbone_range
        self.head = head
        self.fusion_method = fusion_method
        self.esa_arch = esa_arch
        self.context = context
        self.input_size = input_size

        if self.fusion_method == 'fuse':
            self.se_layer0 = SqueezeAndExciteFusionAdd(64)
            self.se_layer1 = SqueezeAndExciteFusionAdd(256)
            self.se_layer2 = SqueezeAndExciteFusionAdd(512)
            self.se_layer3 = SqueezeAndExciteFusionAdd(1024)
            self.se_layer4 = SqueezeAndExciteFusionAdd(2048)
        if self.context:
            self.context = AdaptivePyramidPoolingModule(2048, 2048, input_size=self.input_size)
        if self.esa_arch:
            print("Using the ESANet architecture.")

    def forward(self, input):
        # Forward pass through the RGB backbone
        if self.esa_arch:

            # store the input shape the adaptive pyramid pooling later
            input_shape = input["image"].shape[-2:]

            # layer "0" fusion
            rgb = self.backbone_rgb.conv1(input["image"])
            range = self.backbone_range.conv1(input["range_view"])
            fused = self.se_layer0(rgb, range)

            rgb = self.backbone_rgb.maxpool(fused)
            range = self.backbone_range.maxpool(range)

            # FCN layer 1 fused
            rgb = self.backbone_rgb.layer1(rgb)
            range = self.backbone_range.layer1(range)
            fused = self.se_layer1(rgb, range)

            # FCN layer 2 fused
            rgb = self.backbone_rgb.layer2(fused)
            range = self.backbone_range.layer2(range)
            fused = self.se_layer2(rgb, range)

            # FCN layer 3 fused
            rgb = self.backbone_rgb.layer3(fused)
            range = self.backbone_range.layer3(range)
            fused = self.se_layer3(rgb, range)

            # FCN layer 4 fused
            rgb = self.backbone_rgb.layer4(fused)
            range = self.backbone_range.layer4(range)
            fused_features = self.se_layer4(rgb, range)

            # if self.context:
            # #     fused_features = self.context(fused)
            # else:
            #     pass
        else:
            input_shape = input["image"].shape[-2:]

            rgb_features = self.backbone_rgb(input["image"])

            # Forward pass through the range backbone
            range_features = self.backbone_range(input["range_view"])

        # Fuse the features using the specified method (e.g., addition)
        if self.fusion_method == 'add':
            if isinstance(rgb_features, BaseModelOutputWithPooling):
                fused_features = rgb_features.last_hidden_state[:, 1:, :] + range_features.last_hidden_state[:, 1:, :]
            else:
                fused_features = rgb_features["out"] + range_features["out"]
        elif self.fusion_method == 'fuse':
            pass  # we are doing everything in the ESANet arch
        else:
            # Add other fusion methods as needed
            raise NotImplementedError(f"Fusion method {self.fusion_method} not implemented.")

        # Forward pass through the head
        output = self.head(fused_features)
        output = F.interpolate(output, size=input_shape, mode="bilinear", align_corners=False)

        return output


class Ules(LightningModule):

    def __init__(self, cfg: dict, dataset_name: str, rgb_only_ft: bool, double_backbone: bool, head_only: bool,
                 mean: Optional[List[float]], std: Optional[List[float]], unfreeze_epoch: int, input_size: List[int]):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.n_classes = cfg['data'][dataset_name]['num_classes']
        self.dropout = cfg['model']['dropout']
        self.epochs = cfg['train']['max_epoch']
        self.warmup = cfg['train']['validation_warmup']

        self.lr = cfg['train']['lr']
        self.init = cfg['model']['initialization']

        self.dataset_path = cfg["data"][dataset_name]["location"]

        self.double_backbone = double_backbone
        self.unfreeze_epoch = unfreeze_epoch
        self.input_size = input_size

        # self.sem_loss = mIoULoss([1., 0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
        #                           1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
        #                           1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        sem_loss_weights = [1] * self.n_classes
        self.sem_loss_mIoU = mIoULoss(sem_loss_weights)
        ignore_idx = 0
        if "multi" in self.dataset_path:
            ignore_idx = 255
        if "vis-nir" in self.dataset_path:
            ignore_idx = 255

        class_list = torch.arange(0, self.n_classes)

        self.sem_loss_dice = DiceLoss(mode="multiclass", ignore_index=ignore_idx)
        self.sem_loss_focal = FocalLoss(mode="multiclass", ignore_index=ignore_idx)  #, gamma=2.0)
        self.sem_loss_Lovasz = LovaszLoss(mode="multiclass", ignore_index=ignore_idx)
        self.sem_loss_jaccard = JaccardLoss(mode="multiclass", classes=class_list)

        if "multi" in self.dataset_path:
            self.iou = JaccardIndex(num_classes=self.n_classes, average='none', task="multiclass", ignore_index=255)
        elif "vis-nir" in self.dataset_path:
            self.iou = JaccardIndex(num_classes=self.n_classes, average='none', task="multiclass", ignore_index=255)
        elif "kitti" in self.dataset_path:
            self.iou = JaccardIndex(num_classes=self.n_classes, average='none', task="multiclass", ignore_index=0)
        else:
            self.iou = JaccardIndex(num_classes=self.n_classes, average='none', task="multiclass")

        if "multi" in self.dataset_path:
            self.miou = JaccardIndex(num_classes=self.n_classes, average='weighted', task="multiclass", ignore_index=255)
        elif "vis-nir" in self.dataset_path:
            self.miou = JaccardIndex(num_classes=self.n_classes, average='weighted', task="multiclass", ignore_index=255)
        elif "kitti" in self.dataset_path:
            self.miou = JaccardIndex(num_classes=self.n_classes, average='weighted', task="multiclass", ignore_index=0)
        else:
            self.miou = JaccardIndex(num_classes=self.n_classes, average='weighted', task="multiclass")

        self.accumulated_iou = torch.zeros(self.n_classes).cuda()
        self.accumulated_miou = torch.zeros(1).cuda()

        self.accumulated_miou_loss = 0.0
        self.val_loss = 0.0

        pretrained = False
        dino = True
        if self.double_backbone:
            # TODO remove this pretrained test branch
            if pretrained:
                model_rgb = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=True,
                                                                               aux_loss=None)
                model_rgb.classifier[-1] = torch.nn.Conv2d(512, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
            else:
                model_rgb = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=True,num_classes=self.n_classes,aux_loss=None)
                if dino:
                    id2label = {
                        0: "asphalt",
                        1: "gravel",
                        2: "soil",
                        3: "sand",
                        4: "bush",
                        5: "forest",
                        6: "lowgrass",
                        7: "highgrass",
                        8: "misc.vegetation",
                        9: "treecrown",
                        10: "treetrunk",
                        11: "building",
                        12: "fence",
                        13: "wall",
                        14: "car",
                        15: "bus",
                        16: "sky",
                        17: "misc.object",
                        19: "pole",
                        18: "trafficsign",
                        20: "person",
                        21: "animal",
                        22: "egovehicle",
                        255: "undefined"
                    }
                    model_rgb = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base", id2label=id2label, num_labels=len(id2label))
            model_range = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True,
                                                                             num_classes=self.n_classes,
                                                                             aux_loss=None)
            if dino:
                model_range=  Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base",id2label=id2label,
                                                                              num_labels=len(id2label))



            if dino:
                backbone_rgb = model_rgb.dinov2
                backbone_range = model_range.dinov2
            else:
                backbone_rgb = model_rgb.backbone
                backbone_range = model_range.backbone

            multi1d = True  # temp hardcoded for multimodal dataset
            if multi1d:
                if dino:
                    backbone_range.embeddings.patch_embeddings.projection = torch.nn.Conv2d(1, 768, kernel_size=(14, 14), stride=(14, 14))
                    backbone_range.config.num_channels = 1
                    backbone_range.embeddings.patch_embeddings.num_channels = 1
                else:
                    backbone_range.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7),
                                                       stride=(2, 2), padding=(3, 3), bias=False)

            classifier = model_rgb.classifier
            self.model = DbSegmenter(backbone_rgb, backbone_range, classifier, fusion_method="add", esa_arch=False,
                                     context=False, input_size=self.input_size)
        else:
            print("Instantiating a single backbone model.")
            # self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True,
            #                                                                 num_classes=self.n_classes, aux_loss=None)
            id2label = {
                0: "asphalt",
                1: "gravel",
                2: "soil",
                3: "sand",
                4: "bush",
                5: "forest",
                6: "lowgrass",
                7: "highgrass",
                8: "misc.vegetation",
                9: "treecrown",
                10: "treetrunk",
                11: "building",
                12: "fence",
                13: "wall",
                14: "car",
                15: "bus",
                16: "sky",
                17: "misc.object",
                19: "pole",
                18: "trafficsign",
                20: "person",
                21: "animal",
                22: "egovehicle",
                255: "undefined"
            }
            self.model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base", id2label=id2label, num_labels=len(id2label))
        self.rgb_only_ft = rgb_only_ft
        self.head_only = head_only

        self.mean = mean
        self.std = std

        # self.freeze_backbone()
        # print("backbone has been frozen")

    def freeze_backbone(self):
        if self.double_backbone:
            for param in self.model.backbone_rgb.parameters():
                param.requires_grad = False
            for param in self.model.backbone_range.parameters():
                param.requires_grad = False
        else:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.double_backbone:
            for param in self.model.backbone_rgb.parameters():
                param.requires_grad = True
            for param in self.model.backbone_range.parameters():
                param.requires_grad = True
        else:
            for param in self.model.backbone.parameters():
                param.requires_grad = True

    def forward(self, input):
        if self.double_backbone:
            if self.rgb_only_ft:
                input_shape = input["image"].shape[-2:]
                out = self.model.backbone_rgb(input["image"])
                out = self.model.head(out["out"])
                out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
            else:
                out = self.model(input)
            return out
        else:
            out = self.model(input)
            return out['out']

    def getLoss(self, pred, target, is_train=True):
        # target = target.long()

        if "roses" in self.dataset_path:
            target = target.squeeze()
        # sem_loss = (0.5 * self.sem_loss_Lovasz(pred, target.long()) + 0.3 * self.sem_loss_focal(pred, target.long()) + 0.2 * self.sem_loss_dice(pred, target.long()))
        sem_loss = 0.8 * self.sem_loss_dice(pred, target.long()) + (0.2 * self.sem_loss_focal(pred, target.long()))
        # sem_loss = self.sem_loss_mIoU(pred, target.long())
        # sem_loss = self.sem_loss_Lovasz(pred, target)
        # sem_loss = self.sem_loss_jaccard(pred, target.long())

        if is_train:
            self.accumulated_miou_loss += sem_loss.detach()
        else:
            self.val_loss += sem_loss.detach()

        return sem_loss

    def on_train_epoch_end(self):

        # n_samples = float(len(self.trainer.datamodule.train_dataloader()))
        # # # n_samples = 38
        # #
        # sem_loss = self.accumulated_miou_loss / n_samples
        # #
        # # tensorboard logs
        # self.logger.experiment.add_scalar(
        #     "Loss/sem_loss", sem_loss, self.trainer.current_epoch)
        #
        # self.accumulated_miou_loss *= 0.0
        if self.current_epoch == self.unfreeze_epoch:
            # self.unfreeze_backbone()
            self.configure_optimizers()
            print("backbone has been unfrozen")
        self.training_step_outputs.clear()

    def training_step(self, batch, batch_idx):
        if self.double_backbone:
            x = {"image": batch['image'].float(), "range_view": batch['range_view'].float()}
        else:
            x = batch['image'].float()
        pred = self.forward(x)

        gt = batch['target']
        if "cityscapes" in self.dataset_path:
            gt_encoded = encode_segmap(gt).long()  # only use a subset of classes
        else:
            gt_encoded = gt
        spix_flag = True
        if spix_flag:
            pass
            # from skimage.segmentation import slic
            # for i in range(len(x["image"])):
            # numSegments = 100
            # image_array = np.asarray(pred.permute(1, 2, 0)).astype(np.float64)
            # segments = slic(pred, n_segments=numSegments, sigma=1, compactness=10, enforce_connectivity=True)
            # one_hot_masks, mask_values_list = segment_to_onehot(segments)
            # new_decode = pred.copy()
            # for mask in one_hot_masks[0]:
            #     broadcast_mask = mask[:, :, np.newaxis]
            #     masked_image = decoded_full * broadcast_mask
            #     values, counts = np.unique(masked_image.reshape(-1, masked_image.shape[2]), axis=0, return_counts=True)
            #     counts = counts[1:]  # remove 0 value from masking
            #     if len(counts) == 1:
            #         continue
            #     # plt.imshow(masked_image)
            #     # plt.show()
            #     count_tot = 0
            #     for count in counts:
            #         count_tot += count
            #     threshold = count_tot * 0.7
            #     if counts.max() > threshold:
            #         override_val = values[np.argmax(counts) + 1]
            #         new_decode[mask == 1] = override_val

        loss = self.getLoss(pred, gt_encoded)

        self.log('sem_loss', loss)
        self.training_step_outputs.append(loss)

        # import math
        # loss_x = math.radians(self.trainer.current_epoch*(360/50))
        # loss_modifier = math.cos(loss_x) + 1.1
        return loss
        #return loss * loss_modifier

    def validation_step(self, batch, batch_idx):
        if self.double_backbone:
            x = {"image": batch['image'].float(), "range_view": batch['range_view'].float()}
        else:
            x = batch['image'].float()
        gt = batch['target']

        pred = self.forward(x)
        if "cityscapes" in self.dataset_path:
            gt_encoded = encode_segmap(gt).long()  # only use a subset of classes
        elif "roses" in self.dataset_path:
            gt_encoded = gt.squeeze()
        else:
            gt_encoded = gt

        iou = self.iou(pred, gt_encoded)
        miou = self.miou(pred, gt_encoded)

        _ = self.getLoss(pred, batch['target'], False)

        # sum all the ious because in validation_epoch_end we divide by the number of batches = mean over validation set
        self.accumulated_iou += iou
        self.accumulated_miou += miou

        # Log images every at start and end epoch
        # if self.trainer.current_epoch == 0 or self.trainer.current_epoch == 99:
        for img_idx, img_tuple in enumerate(zip(batch['image'], gt_encoded, pred)):
            # store images in variables
            # input_img = img_tuple[0].detach().cpu()
            input_img = img_tuple[0][0:3, ...].detach().cpu()

            if self.mean:
                mean = tuple(self.mean)
            else:
                mean = (0, 0, 0)
            if self.std:
                std = tuple(self.std)
            else:
                std = (1, 1, 1)
            # decode all gt
            if "cityscapes" in self.dataset_path:
                ground_truth = decode_segmap(img_tuple[1][..., 0:3, ...].detach().cpu())
            elif "kitti" in self.dataset_path:
                ground_truth = kitti_decode(img_tuple[1].detach().cpu())
            elif "multimodal" in self.dataset_path:
                ground_truth = multi_mat_decode(img_tuple[1].detach().cpu())
            elif "vis-nir" in self.dataset_path:
                ground_truth = visnir_decode(img_tuple[1].detach().cpu())
            elif "roses" in self.dataset_path:
                ground_truth = img_tuple[1].detach().cpu()

            # decode preds as well
            encoded_pred = torch.argmax(torch.softmax(img_tuple[2], 0), 0)
            if "cityscapes" in self.dataset_path:
                decoded_pred = decode_segmap(encoded_pred.detach().cpu())
            elif "kitti" in self.dataset_path:
                decoded_pred = kitti_decode(encoded_pred.detach().cpu())
                mean = (0.35095342, 0.36734804, 0.36330285)
                std = (0.30601038, 0.31168418, 0.32000023)
            elif "multimodal" in self.dataset_path:
                decoded_pred = multi_mat_decode(encoded_pred.detach().cpu())
                # mean = (0.485, 0.456, 0.406)
                # std = (0.229, 0.224, 0.225)
            elif "vis-nir" in self.dataset_path:
                decoded_pred = visnir_decode((encoded_pred.detach().cpu()))
            elif "roses" in self.dataset_path:
                decoded_pred = encoded_pred.detach().cpu()

            # put them in a plot, so they are all printed together
            fig, ax = plt.subplots(ncols=3)
            denormalized_img = input_img * np.array(std)[:, None, None] + np.array(mean)[:, None, None]
            ax[0].imshow(np.moveaxis(denormalized_img.numpy(), 0, 2))
            ax[1].imshow(ground_truth)
            ax[2].imshow(decoded_pred)
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            ax[0].set_title('Input Image')
            ax[1].set_title('Ground truth')
            ax[2].set_title('Prediction')
            plt.tight_layout()
            plt.margins(x=0)
            plt.margins(y=0)
            self.logger.experiment.add_figure(f"Results/{batch_idx}_{img_idx}", fig)

    def on_validation_epoch_end(self):
        n_batches = len(self.trainer.datamodule.val_dataloader())
        n_samples = float(len(self.trainer.datamodule.val_dataloader().dataset))

        self.accumulated_iou /= n_batches
        self.accumulated_miou /= n_batches
        self.val_loss /= n_samples

        if "cityscapes" in self.dataset_path:
            self.logger.experiment.add_scalars(
                "Metrics_iou", {'unlabelled': self.accumulated_iou[0], 'road': self.accumulated_iou[1],
                                'sidewalk': self.accumulated_iou[2], 'building': self.accumulated_iou[3],
                                'wall': self.accumulated_iou[4], 'fence': self.accumulated_iou[5],
                                'pole': self.accumulated_iou[6], 'traffic_light': self.accumulated_iou[7],
                                'traffic_sign': self.accumulated_iou[8], 'vegetation': self.accumulated_iou[9],
                                'terrain': self.accumulated_iou[10], 'sky': self.accumulated_iou[11],
                                'person': self.accumulated_iou[12], 'rider': self.accumulated_iou[13],
                                'car': self.accumulated_iou[14], 'truck': self.accumulated_iou[15],
                                'bus': self.accumulated_iou[16], 'train': self.accumulated_iou[17],
                                'motorcycle': self.accumulated_iou[18], 'bicycle': self.accumulated_iou[19]},
                self.trainer.current_epoch)
        elif "kitti" in self.dataset_path:
            self.logger.experiment.add_scalars(
                "Metrics_iou", {'unlabelled': self.accumulated_iou[0], 'building': self.accumulated_iou[1],
                                'road': self.accumulated_iou[2], 'sidewalk': self.accumulated_iou[3],
                                'fence': self.accumulated_iou[4], 'vegetation': self.accumulated_iou[5],
                                'pole': self.accumulated_iou[6], 'car': self.accumulated_iou[7],
                                'sign': self.accumulated_iou[8], 'pedestrian': self.accumulated_iou[9],
                                'cyclist': self.accumulated_iou[10], 'sky': self.accumulated_iou[11]},
                self.trainer.current_epoch)
        elif "multi" in self.dataset_path:
            self.logger.experiment.add_scalars(
                "Metrics_iou", {'asphalt': self.accumulated_iou[0], 'concrete': self.accumulated_iou[1],
                                'metal': self.accumulated_iou[2], 'road_marking': self.accumulated_iou[3],
                                'fabric': self.accumulated_iou[4], 'glass': self.accumulated_iou[5],
                                'plaster': self.accumulated_iou[6], 'plastic': self.accumulated_iou[7],
                                'rubber': self.accumulated_iou[8], 'sand': self.accumulated_iou[9],
                                'gravel': self.accumulated_iou[10], 'ceramic': self.accumulated_iou[11],
                                'cobblestone': self.accumulated_iou[12], 'brick': self.accumulated_iou[13],
                                'grass': self.accumulated_iou[14], 'wood': self.accumulated_iou[15],
                                'leaf': self.accumulated_iou[16], 'water': self.accumulated_iou[17],
                                'human': self.accumulated_iou[18], 'sky': self.accumulated_iou[19]},
                self.trainer.current_epoch)
        elif "vis-nir" in self.dataset_path:
            self.logger.experiment.add_scalars(
                "Metrics_iou", {'asphalt': self.accumulated_iou[0], 'gravel': self.accumulated_iou[1],
                                'soil': self.accumulated_iou[2], 'sand': self.accumulated_iou[3],
                                'bush': self.accumulated_iou[4], 'forest': self.accumulated_iou[5],
                                'grass_l': self.accumulated_iou[6], 'grass_h': self.accumulated_iou[7],
                                'veg': self.accumulated_iou[8], 'tree_c': self.accumulated_iou[9],
                                'tree_t': self.accumulated_iou[10], 'building': self.accumulated_iou[11],
                                'fence': self.accumulated_iou[12], 'wall': self.accumulated_iou[13],
                                'car': self.accumulated_iou[14], 'bus': self.accumulated_iou[15],
                                'sky': self.accumulated_iou[16], 'misc': self.accumulated_iou[17],
                                'pole': self.accumulated_iou[18], 'sign': self.accumulated_iou[19],
                                'person': self.accumulated_iou[20], 'animal': self.accumulated_iou[21],
                                'ego_vehicle': self.accumulated_iou[22]},
                self.trainer.current_epoch)
        elif "roses" in self.dataset_path:
            self.logger.experiment.add_scalars(
                "Metrics_iou", {'ground': self.accumulated_iou[0], 'weed': self.accumulated_iou[1],
                                'crop': self.accumulated_iou[2]},
                self.trainer.current_epoch)

        self.logger.experiment.add_scalar(
            "mean_IoU", self.accumulated_miou, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "Loss/val_total_loss", self.val_loss, self.trainer.current_epoch)

        self.accumulated_iou *= 0
        self.accumulated_miou *= 0
        self.val_loss *= 0
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x = batch['image'].float()
        gt = batch['target']

        pred = self.forward(x)
        if "cityscapes" in self.dataset_path:
            gt_encoded = encode_segmap(gt).long()  # only use a subset of classes
        elif "kitti" in self.dataset_path:
            gt_encoded = gt  # only use a subset of classes
        iou = self.iou(pred, gt_encoded)
        miou = self.miou(pred, gt_encoded)

        # sum all the ious because in validation_epoch_end we divide by the number of batches = mean over validation set
        self.accumulated_iou += iou
        self.accumulated_miou += miou

    def on_test_epoch_end(self):
        n_batches = len(self.trainer.datamodule.test_dataloader())
        self.accumulated_iou /= n_batches
        self.accumulated_miou /= n_batches

        self.logger.experiment.add_scalars(
            "Metrics_iou", {'unlabelled': self.accumulated_iou[0], 'road': self.accumulated_iou[1],
                            'sidewalk': self.accumulated_iou[2], 'building': self.accumulated_iou[3],
                            'wall': self.accumulated_iou[4], 'fence': self.accumulated_iou[5],
                            'pole': self.accumulated_iou[6], 'traffic_light': self.accumulated_iou[7],
                            'traffic_sign': self.accumulated_iou[8], 'vegetation': self.accumulated_iou[9],
                            'terrain': self.accumulated_iou[10], 'sky': self.accumulated_iou[11],
                            'person': self.accumulated_iou[12], 'rider': self.accumulated_iou[13],
                            'car': self.accumulated_iou[14], 'truck': self.accumulated_iou[15],
                            'bus': self.accumulated_iou[16], 'train': self.accumulated_iou[17],
                            'motorcycle': self.accumulated_iou[18], 'bicycle': self.accumulated_iou[19]},
            self.trainer.current_epoch)
        self.logger.experiment.add_scalar(
            "mean_IoU", self.accumulated_miou, self.current_epoch
        )

        self.accumulated_iou *= 0
        self.accumulated_miou *= 0

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch["image"])

    def configure_optimizers(self):
        #self.optimizers_backbone_rgb = torch.optim.AdamW(
        #    self.model.backbone_rgb.parameters(), lr=self.lr[0]*0.1)
        #self.optimizers_backbone_range = torch.optim.AdamW(f
        #    self.model.backbone_range.parameters(), lr=self.lr[0]*0.1)
        if self.head_only and self.double_backbone:
            self.optimizers = torch.optim.AdamW(self.model.head.parameters(), lr=self.lr[0])
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizers,
                max_lr=5e-4,  # Corresponding to the two parameter groups
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=25.0,
                final_div_factor=10000.0,
                three_phase=False
            )

            return {
                "optimizer": self.optimizers,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif self.head_only and not self.double_backbone:
            self.optimizers = torch.optim.AdamW(self.model.classifier.parameters(), lr=self.lr[0])
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizers,
                max_lr=5e-4,  # Corresponding to the two parameter groups
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=25.0,
                final_div_factor=10000.0,
                three_phase=False
            )

            return {
                "optimizer": self.optimizers,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif not self.head_only and self.double_backbone:
            self.optimizers = torch.optim.AdamW([
                {'params': self.model.backbone_rgb.parameters(), 'lr': self.lr[0]},
                {'params': self.model.backbone_range.parameters(), 'lr': self.lr[0]},
                {'params': self.model.head.parameters(), 'lr': self.lr[1]}
            ])

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizers,
                max_lr=[5e-4, 5e-4, 5e-3],  # Corresponding to the two parameter groups
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=25.0,
                final_div_factor=10000.0,
                three_phase=False
            )

            return {
                "optimizer": self.optimizers,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif not self.head_only and not self.double_backbone:

            self.optimizers = torch.optim.AdamW(self.model.parameters(), lr=self.lr[0])
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizers,
                max_lr=5e-4,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=25.0,
                final_div_factor=10000.0,
                three_phase=False
            )

            return {
                "optimizer": self.optimizers,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        # return [self.optimizers]


def main():
    import yaml
    cfg = yaml.safe_load(open("/home/matt/PycharmProjects/ULES/config/config.yaml"))
    dataset_name = "KyotoMaterialSeg"

    if dataset_name in cfg["data"].keys():
        image_size = cfg["data"][dataset_name]['image_size']
        mean = cfg["data"][dataset_name]['mean']
        std = cfg["data"][dataset_name]['std']
    else:
        raise Exception("No dataset named {}".format(dataset_name))

    # Load data and model
    mean = None
    std = None
    head_only = False
    rgb_only_ft = False
    double_backbone = True
    model = Ules(cfg, dataset_name, rgb_only_ft, double_backbone, head_only, mean, std, unfreeze_epoch=200,
                 input_size=[260, 260])

    print(model)

    model.eval()
    sample = {"image": torch.randn(1, 3, image_size[0], image_size[1]),
              "range_view": torch.randn(1, 1, image_size[0], image_size[1])}

    with torch.no_grad():
        output = model(sample)
    print(output.shape)


if __name__ == '__main__':
    main()
