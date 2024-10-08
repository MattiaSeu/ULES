'''
Unsupervised Learning Enhanced Segmentation
'''
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from pytorch_lightning import LightningModule
from models.blocks import DownsamplerBlock, UpsamplerBlock, non_bottleneck_1d
from models.loss import mIoULoss, BinaryFocalLoss, CrossEntropyLoss, AsymmetricUnifiedFocalLoss
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt
from utils.segmap import encode_segmap, decode_segmap, kitti_encode, kitti_decode, multi_mat_decode, visnir_decode
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss, JaccardLoss, FocalLoss

cityscapes_idx2class_map = ['unlabelled', 'road',
                 'sidewalk', 'building',
                 'wall', 'fence',
                 'pole', 'traffic_light',
                 'traffic_sign', 'vegetation',
                 'terrain', 'sky',
                 'person', 'rider',
                 'car', 'truck',
                 'bus', 'train',
                 'motorcycle', 'bicycle']


class Ules(LightningModule):

    def __init__(self, cfg: dict, mean: Optional[List[float]], std: Optional[List[float]]):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.n_classes = cfg['tasks']['semantic_segmentation']['n_classes']
        self.dropout = cfg['model']['dropout']
        self.epochs = cfg['train']['max_epoch']
        self.warmup = cfg['train']['validation_warmup']

        self.lr = cfg['train']['lr']
        self.init = cfg['model']['initialization']

        self.dataset_path = cfg["data"]["ft-path"]

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
        self.sem_loss_focal = FocalLoss(mode="multiclass", ignore_index=ignore_idx) #, gamma=2.0)
        self.sem_loss_Lovasz = LovaszLoss(mode="multiclass", ignore_index=ignore_idx)
        self.sem_loss_jaccard = JaccardLoss(mode="multiclass", classes=class_list)



        if "multi" in self.dataset_path:
            self.iou = JaccardIndex(num_classes=self.n_classes, average='none', task="multiclass", ignore_index=255)
        elif "vis-nir" in self.dataset_path:
            self.iou = JaccardIndex(num_classes=self.n_classes, average='none', task="multiclass", ignore_index=255)
        else:
            self.iou = JaccardIndex(num_classes=self.n_classes, average='none', task="multiclass")

        if "multi" in self.dataset_path:
            self.miou = JaccardIndex(num_classes=self.n_classes, average='weighted', task="multiclass", ignore_index=255)
        elif "vis-nir" in self.dataset_path:
            self.miou = JaccardIndex(num_classes=self.n_classes, average='weighted', task="multiclass", ignore_index=255)
        else:
            self.miou = JaccardIndex(num_classes=self.n_classes, average='weighted', task="multiclass")

        self.accumulated_iou = torch.zeros(self.n_classes).cuda()
        self.accumulated_miou = torch.zeros(1).cuda()

        self.accumulated_miou_loss = 0.0
        self.val_loss = 0.0

        self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=self.n_classes,
                                                                  aux_loss=None)
        self.mean = mean
        self.std = std
        # self.model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, input):
        out = self.model(input)
        return out['out']

    def getLoss(self, pred, target, is_train=True):
        # target = target.long()

        if "roses" in self.dataset_path:
            target = target.squeeze()
        sem_loss = self.sem_loss_dice(pred, target.long()) + (0.2 * self.sem_loss_focal(pred, target))
        # sem_loss = self.sem_loss_mIoU(pred, target.long())
        # sem_loss = self.sem_loss_Lovasz(pred, target)
        # sem_loss = self.sem_loss_jaccard(pred, target.long())

        if is_train:
            self.accumulated_miou_loss += sem_loss.detach()
        else:
            self.val_loss += sem_loss.detach()

        return sem_loss

    def on_train_epoch_end(self):

        # n_samples = float(len(self.train_dataloader()))

        # n_samples = float(len(self.trainer.datamodule.train_dataloader()))
        # # n_samples = 38
        #
        # sem_loss = self.accumulated_miou_loss / n_samples
        #
        # # tensorboard logs
        # self.logger.experiment.add_scalar(
        #     "Loss/sem_loss", sem_loss, self.trainer.current_epoch)
        #
        # self.accumulated_miou_loss *= 0.0
        self.training_step_outputs.clear()

    def training_step(self, batch, batch_idx):
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
        return loss

    def validation_step(self, batch, batch_idx):
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
                "Metrics_iou", {'asphalt'      : self.accumulated_iou[0],  'concrete'    : self.accumulated_iou[1],
                                'metal'        : self.accumulated_iou[2],  'road_marking': self.accumulated_iou[3],
                                'fabric'       : self.accumulated_iou[4],  'glass'       : self.accumulated_iou[5],
                                'plaster'      : self.accumulated_iou[6],  'plastic'     : self.accumulated_iou[7],
                                'rubber'       : self.accumulated_iou[8],  'sand'        : self.accumulated_iou[9],
                                'gravel'       : self.accumulated_iou[10], 'ceramic'    : self.accumulated_iou[11],
                                'cobblestone'  : self.accumulated_iou[12], 'brick'      : self.accumulated_iou[13],
                                'grass'        : self.accumulated_iou[14], 'wood'       : self.accumulated_iou[15],
                                'leaf'         : self.accumulated_iou[16], 'water'      : self.accumulated_iou[17],
                                'human'        : self.accumulated_iou[18], 'sky'        : self.accumulated_iou[19]},
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

    def configure_optimizers(self):
        # self.optimizers = torch.optim.AdamW(
        #     self.model.classifier.parameters(), lr=self.lr[0])
        self.optimizers = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr[0])

        return [self.optimizers]


class ERFNetEncoder(nn.Module):

    def __init__(self, num_classes, dropout=0.1, batch_norm=True, instance_norm=False, init=None):
        super().__init__()

        self.initial_block = DownsamplerBlock(3, 16, batch_norm, instance_norm, init)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64, batch_norm, instance_norm, init))

        DROPOUT = dropout
        for x in range(0, 10):  # 5 times
            self.layers.append(non_bottleneck_1d(64, DROPOUT, 1, batch_norm, instance_norm, init))

        self.layers.append(DownsamplerBlock(64, 128, batch_norm, instance_norm, init))

        DROPOUT = dropout
        for x in range(0, 3):  # 2 times
            self.layers.append(non_bottleneck_1d(128, DROPOUT, 2, batch_norm, instance_norm, init))
            self.layers.append(non_bottleneck_1d(128, DROPOUT, 4, batch_norm, instance_norm, init))
            self.layers.append(non_bottleneck_1d(128, DROPOUT, 8, batch_norm, instance_norm, init))
            self.layers.append(non_bottleneck_1d(128, DROPOUT, 16, batch_norm, instance_norm, init))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(
            128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = []
        output.append(self.initial_block(input))

        for layer in self.layers:
            output.append(layer(output[-1]))

        if predict:
            output.append(self.output_conv(output[-1]))

        return output


class DecoderSemanticSegmentation(nn.Module):
    def __init__(self, num_classes: int, dropout: float, batch_norm=True, instance_norm=False, init=None):
        super().__init__()
        self.dropout = dropout

        self.layers1 = nn.Sequential(
            UpsamplerBlock(128, 64, init),
            non_bottleneck_1d(64, self.dropout, 1, batch_norm, instance_norm, init),
            non_bottleneck_1d(64, self.dropout, 1, batch_norm, instance_norm, init),
        )

        self.layers2 = nn.Sequential(
            UpsamplerBlock(64, 16, init),
            non_bottleneck_1d(16, self.dropout, 1, batch_norm, instance_norm, init),
            non_bottleneck_1d(16, self.dropout, 1, batch_norm, instance_norm, init)
        )

        self.output_conv = nn.ConvTranspose2d(
            in_channels=16, out_channels=num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, input):
        # skip2, _, _, _, _, _, skip1, _, _, _, _, _, _, _, _, out = input

        output1 = self.layers1(input[-1])  # + skip1
        output2 = self.layers2(output1)  # + skip2
        out = self.output_conv(output2)

        return out, [output2, output1]
