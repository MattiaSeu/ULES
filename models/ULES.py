'''
Unsupervised Learning Enhanced Segmentation
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
from models.blocks import DownsamplerBlock, UpsamplerBlock, non_bottleneck_1d
from models.loss import mIoULoss, BinaryFocalLoss
from torchmetrics import IoU
import matplotlib.pyplot as plt
from utils.segmap import encode_segmap, decode_segmap


class Ules(LightningModule):

    def __init__(self, cfg: dict):
        super().__init__()
        self.n_classes = cfg['tasks']['semantic_segmentation']['n_classes']
        self.dropout = cfg['model']['dropout']
        self.epochs = cfg['train']['max_epoch']
        self.warmup = cfg['train']['validation_warmup']

        self.lr = cfg['train']['lr']
        self.init = cfg['model']['initialization']

        self.sem_loss = mIoULoss([1., 0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                                  1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                                  1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.iou = IoU(num_classes=20, reduction='none')
        self.accumulated_miou = torch.zeros(20).cuda()

        self.accumulated_iou_loss = 0.0
        self.val_loss = 0.0

        # encoder
        self.encoder = ERFNetEncoder(self.n_classes, dropout=self.dropout, init=self.init)

        # decoders
        self.decoder_semseg = DecoderSemanticSegmentation(self.n_classes, self.dropout, init=self.init)

    def forward(self, input):
        # encoder -- possibly from pre-trained weights
        out = self.encoder(input)

        # encoder_lr = self.scheduler_encoder.get_last_lr()
        # self.logger.experiment.add_scalar("LR/encoder", encoder_lr[0], self.trainer.current_epoch)

        # semantic segmentation
        semantic, skips = self.decoder_semseg(out)

        return semantic

    def getLoss(self, pred, target, is_train=True):
        # target = target.long()

        sem_loss = self.sem_loss(pred, target)

        if is_train:
            self.accumulated_iou_loss += sem_loss.detach()
        else:
            self.val_loss += sem_loss.detach()

        return sem_loss

    def training_epoch_end(self, training_step_outputs):

        n_samples = float(len(self.train_dataloader()))

        sem_loss = self.accumulated_iou_loss / n_samples

        # tensorboard logs
        self.logger.experiment.add_scalar(
            "Loss/sem_loss", sem_loss, self.trainer.current_epoch)

        self.accumulated_iou_loss *= 0.0

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch['image'].float()
        sem = self.forward(x)
        target = encode_segmap(batch['target'])
        loss = self.getLoss(sem, target)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image'].float()
        p_labels = batch['target']

        pred = self.forward(x)
        gt_batch = encode_segmap(p_labels).long()
        iou = self.iou(pred, gt_batch)

        _ = self.getLoss(pred, batch['target'], False)

        # we sum all the ious, because in validation_epoch_end we divide by the number of samples = mean over validation set
        self.accumulated_miou += iou

        if True:  # self.trainer.current_epoch >= self.warmup or (self.trainer.current_epoch % 30 == 0 and self.trainer.current_epoch > 70):
            for item in range(pred.shape[0]):
                semantic = torch.argmax(torch.softmax(pred[item], 0), 0)

    def validation_epoch_end(self, validation_step_outputs):
        n_batches = len(self.val_dataloader())
        n_samples = float(len(self.val_dataloader().dataset))

        self.accumulated_miou /= n_batches
        self.val_loss /= n_samples

        self.logger.experiment.add_scalars(
            "Metrics/iou", {'soil': self.accumulated_miou[0], 'plants': self.accumulated_miou[1]},
            self.trainer.current_epoch)
        self.logger.experiment.add_scalar(
            "Loss/val_total_loss", self.val_loss, self.trainer.current_epoch)

        self.accumulated_miou *= 0
        self.val_loss *= 0

    def configure_optimizers(self):
        # OPTIMIZERS
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(), lr=self.lr[0])
        self.semantic_optimizer = torch.optim.AdamW(
            self.decoder_semseg.parameters(), lr=self.lr[1])
        # SCHEDULERS
        self.scheduler_encoder = torch.optim.lr_scheduler.StepLR(
            self.encoder_optimizer, step_size=25, gamma=0.9)

        return [self.encoder_optimizer, self.semantic_optimizer], [self.scheduler_encoder]


class ERFNetEncoder(nn.Module):

    def __init__(self, num_classes, dropout=0.1, batch_norm=True, instance_norm=False, init=None):
        super().__init__()

        self.initial_block = DownsamplerBlock(3, 16, batch_norm, instance_norm, init)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64, batch_norm, instance_norm, init))

        DROPOUT = dropout
        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, DROPOUT, 1, batch_norm, instance_norm, init))

        self.layers.append(DownsamplerBlock(64, 128, batch_norm, instance_norm, init))

        DROPOUT = dropout
        for x in range(0, 2):  # 2 times
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
            non_bottleneck_1d(64, self.dropout, 1, batch_norm, instance_norm, init))

        self.layers2 = nn.Sequential(
            UpsamplerBlock(64, 16, init),
            non_bottleneck_1d(16, self.dropout, 1, batch_norm, instance_norm, init))

        self.output_conv = nn.ConvTranspose2d(
            in_channels=16, out_channels=num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, input):
        skip2, _, _, _, _, _, skip1, _, _, _, _, _, _, _, _, out = input

        output1 = self.layers1(out) + skip1
        output2 = self.layers2(output1) + skip2
        out = self.output_conv(output2)

        return out, [output2, output1]
