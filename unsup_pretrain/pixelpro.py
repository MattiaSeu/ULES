import torch
import os
import pytorch_lightning as pl
from pixel_level_contrastive_learning.pixel_level_contrastive_learning import PixelCL
from pixel_level_contrastive_learning.pixel_level_contrastive_learning_DB import PixelCL_DB
from torchvision import models
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
# from data_loading.NuScenes import NuScenes
from data_loading.Kitti_DB import KittiRangeDataset
from data_loading.multimodal import MultimodalMaterialDataset
from data_loading.visnir import VisNirDataset
from data_loading.multispecweed import MultiSpectralCropWeed
from utils.tenprint import print_tensor
import click


class LightningPixelCL(pl.LightningModule):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def forward(self, x):
        # print_tensor(x["image"][0])
        # print_tensor(x["range_view"][0])
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs = batch
        outputs = self(inputs)
        self.log_dict(outputs)
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
              default=2)
@click.option('--num_workers',
              '-nw',
              help='number of workers for data loading (tentative equal to cpu cores)',
              default=4)
@click.option('--gpus',
              '-g',
              help='number of gpus to be used',
              default=1)
def main(data_path, extra, checkpoint, batch_size, num_workers, gpus):
    resnet = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=23,
                                              aux_loss=None)
    # resnet.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


    data_path = "/home/matt/data/vis-nir/"
    # checkpoint = "pixpro_crop_db.ckpt"
    # city_data_path = os.path.join(data_path, 'cityscapes/')

    # extra = False
    # image_size = [76, 248]
    # image_size = [130, 130] # multi-modal size
    # image_size = [90, 160]
    image_size = [80, 200]  # 6th of cropped nir image
    # image_size = [106, 160]  # 7th of multispectral image
    learner = PixelCL_DB(
        resnet.backbone,
        image_size=image_size,
        hidden_layer_pixel='layer4',  # leads to output of 8x8 feature map for pixel-level learning
        hidden_layer_instance=-2,  # leads to output for instance-level learning
        projection_size=256,  # size of projection output, 256 was used in the paper
        projection_hidden_size=256,  # size of projection hidden dimension, paper used 2048
        moving_average_decay=0.99,  # exponential moving average decay of target encoder
        ppm_num_layers=1,  # number of layers for transform function in the pixel propagation module, 1 was optimal
        ppm_gamma=2,  # sharpness of the similarity in the pixel propagation module, already at optimal value of 2
        distance_thres=0.7,
        # ideal value is 0.7, as indicated in the paper, which makes the assumption of each feature map's pixel
        # diagonal distance to be 1 (still unclear)
        similarity_temperature=0.3,  # temperature for the cosine similarity for the pixel contrastive loss
        alpha=1.,  # weight of the pixel propagation loss (pixpro) vs pixel CL loss
        use_pixpro=True,  # do pixel pro instead of pixel contrast loss, defaults to pixpro, since it is the best one
        cutout_ratio_range=(0.6, 0.8),  # a random ratio is selected from this range for the random cutout
        use_range_image=True
    )

    opt = torch.optim.AdamW(learner.parameters(), lr=1e-4)
    # opt = torch.optim.SGD(learner.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1)

    model = LightningPixelCL(learner, opt)

    version_name = "visnir3_db"
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=2, monitor="loss",
                                          every_n_epochs=25, filename="{epoch}-%s" % version_name)
    torch.set_float32_matmul_precision('high')

    tb_logger = TensorBoardLogger(save_dir=os.getcwd(), name="unsup_logs", version=version_name)

    trainer = pl.Trainer(devices=gpus, logger=tb_logger, callbacks=[checkpoint_callback],
                         max_epochs=200, accumulate_grad_batches=8)

    if extra:
        split_train = 'train_extra'
        mode = 'coarse'
    else:
        split_train = 'train'
        mode = 'fine'

    # train_data = CityDataPixel(city_data_path, split=split_train, mode=mode, target_type='semantic', transforms=None)
    # val_data = CityDataPixel(city_data_path, split='val', mode=mode, target_type='semantic', transforms=None)
    # train_data = KittiRangeDataset(root_dir=data_path, split="train", image_size=image_size)
    # val_data = KittiRangeDataset(root_dir=data_path, split="test", image_size=image_size)
    # train_data = MultimodalMaterialDataset(root_dir=data_path, split="train", image_size=image_size)
    # val_data = MultimodalMaterialDataset(root_dir=data_path, split="test", image_size=image_size)

    data = VisNirDataset(root_dir=data_path, image_size=image_size)
    #
    # data = MultiSpectralCropWeed(root_dir=data_path, image_size=image_size, split="roses23")
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import train_test_split

    TEST_SIZE = 0.2
    SEED = 42

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices = train_test_split(
        range(len(data)),
        test_size=TEST_SIZE,
        random_state=SEED
    )

    # generate subset based on indices
    train_data = Subset(data, train_indices)



    # val_data = NuScenes()

    batch_size = batch_size

    trainer.fit(model,
                DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                           drop_last=True),
                ckpt_path=checkpoint)

    trainer.save_checkpoint("pixpro_" + version_name + ".ckpt")


if __name__ == '__main__':
    main()
