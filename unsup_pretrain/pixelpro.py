import torch
import os
import pytorch_lightning as pl
from pixel_level_contrastive_learning.pixel_level_contrastive_learning import PixelCL
from torchvision import models
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loading import CityDataPixel
from torch.utils.data import DataLoader
import click


class LightningPixelCL(pl.LightningModule):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs = batch
        outputs = self(inputs)
        return outputs  # The loss is already computed in the forward pass

    def configure_optimizers(self):
        return self.optimizer


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
              help='number of workers for data loading (tent-ative equal to cpu cores)',
              default=12)
def main(data_path, extra, checkpoint, batch_size, num_workers):
    resnet = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=20,
                                              aux_loss=None)
    city_data_path = os.path.join(data_path, 'cityscapes/')

    learner = PixelCL(
        resnet,
        image_size=[128, 256],
        hidden_layer_pixel='classifier',  # leads to output of 8x8 feature map for pixel-level learning
        hidden_layer_instance=-2,  # leads to output for instance-level learning
        projection_size=128,  # size of projection output, 256 was used in the paper
        projection_hidden_size=128,  # size of projection hidden dimension, paper used 2048
        moving_average_decay=0.99,  # exponential moving average decay of target encoder
        ppm_num_layers=1,  # number of layers for transform function in the pixel propagation module, 1 was optimal
        ppm_gamma=2,  # sharpness of the similarity in the pixel propagation module, already at optimal value of 2
        distance_thres=0.7,
        # ideal value is 0.7, as indicated in the paper, which makes the assumption of each feature map's pixel
        # diagonal distance to be 1 (still unclear)
        similarity_temperature=0.3,  # temperature for the cosine similarity for the pixel contrastive loss
        alpha=1.,  # weight of the pixel propagation loss (pixpro) vs pixel CL loss
        use_pixpro=True,  # do pixel pro instead of pixel contrast loss, defaults to pixpro, since it is the best one
        cutout_ratio_range=(0.6, 0.8)  # a random ratio is selected from this range for the random cutout
    )

    opt = torch.optim.AdamW(learner.parameters(), lr=1e-4)

    model = LightningPixelCL(learner, opt)

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=2, monitor="loss", save_last=True,
                                          every_n_epochs=5)
    trainer = pl.Trainer(accelerator="auto", callbacks=[checkpoint_callback], resume_from_checkpoint=checkpoint)

    if extra:
        split_train = 'train_extra'
        mode = 'coarse'
    else:
        split_train = 'train'
        mode = 'fine'

    train_data = CityDataPixel(city_data_path, split=split_train, mode=mode, target_type='semantic', transforms=None)
    val_data = CityDataPixel(city_data_path, split='val', mode=mode, target_type='semantic', transforms=None)

    batch_size = batch_size

    trainer.fit(model,
                DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
                DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
                )


if __name__ == '__main__':
    main()
