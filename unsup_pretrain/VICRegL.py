import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torchvision.datasets import Cityscapes
import click
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from lightly.loss import VICRegLLoss

## The global projection head is the same as the Barlow Twins one
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjectionHead
from lightly.transforms.vicregl_transform import VICRegLTransform


class VICRegL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50()
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection_head = BarlowTwinsProjectionHead(2048, 4096, 4096)
        self.local_projection_head = VicRegLLocalProjectionHead(2048, 512, 512)
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.criterion = VICRegLLoss()

    def forward(self, x):
        x = self.backbone(x)
        y = self.average_pool(x).flatten(start_dim=1)
        z = self.projection_head(y)
        y_local = x.permute(0, 2, 3, 1)  # (B, D, W, H) to (B, W, H, D)
        z_local = self.local_projection_head(y_local)
        return z, z_local

    def training_step(self, batch, batch_index):
        views_and_grids = batch[0]
        views = views_and_grids[: len(views_and_grids) // 2]
        grids = views_and_grids[len(views_and_grids) // 2:]
        features = [self.forward(view) for view in views]
        loss = self.criterion(
            global_view_features=features[:2],
            global_view_grids=grids[:2],
            local_view_features=features[2:],
            local_view_grids=grids[2:],
        )
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        return optim


@click.command()
@click.option('--data_path',
              '-d',
              type=str,
              help='path to the dataset folder',
              default="~/data")
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
@click.option('--extra',
              '-e',
              is_flag=True,
              show_default=True,
              help='use the extra coarse data',
              default=True)
@click.option('--batch_size',
              '-b',
              help='batch size',
              default=16)
@click.option('--num_workers',
              '-nw',
              help='number of workers for data loading (tentative equal to cpu cores)',
              default=12)
@click.option('--gpus',
              '-g',
              help='number of gpus to be used',
              default=1)
def main(data_path, extra, checkpoint, batch_size, num_workers, gpus):
    city_data_path = os.path.join(data_path, 'cityscapes/')
    model = VICRegL()
    transform = VICRegLTransform(n_local_views=0)

    if extra:
        split_train = 'train_extra'
        mode = 'coarse'
    else:
        split_train = 'train'
        mode = 'fine'

    train_data = Cityscapes(city_data_path, split=split_train, mode=mode, target_type='semantic', transform=transform,
                            target_transform=lambda t: 0, )
    val_data = Cityscapes(city_data_path, split='val', mode=mode, target_type='semantic', transform=transform,
                          target_transform=lambda t: 0, )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=2, monitor="loss", save_last=True,
                                          every_n_epochs=10)

    trainer = pl.Trainer(max_epochs=100, devices=gpus, accelerator=accelerator, callbacks=[checkpoint_callback])
    trainer.fit(model=model,
                train_dataloaders=DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                             pin_memory=True,
                                             drop_last=True),
                val_dataloaders=DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                           pin_memory=True,
                                           drop_last=True),
                ckpt_path=checkpoint)
