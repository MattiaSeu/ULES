import torch
import os
import re
import pytorch_lightning as pl
import yaml
from os.path import join, dirname, abspath
from pixel_level_contrastive_learning.pixel_level_contrastive_learning import PixelCL
from pixel_level_contrastive_learning.pixel_level_contrastive_learning_DB import PixelCL_DB
from torchvision import models
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
# from data_loading.NuScenes import NuScenes
from data_loading.Kitti_DB import KittiRangeDataset
from data_loading.multimodal import MultimodalMaterialDataset
from data_loading.visnir import VisNirDataset
from data_loading.multispecweed import MultiSpectralCropWeed
from utils.tenprint import print_tensor
import click
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from pytorch_lightning.plugins import MixedPrecisionPlugin


class LightningPixelCL(pl.LightningModule):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.training_step_outputs = []

    def forward(self, x):
        # print_tensor(x["image"][0])
        # print_tensor(x["range_view"][0])
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs = batch
        outputs = self(inputs)
        self.log_dict(outputs)
        self.training_step_outputs.append(outputs)
        return outputs  # The loss is already computed in the forward pass

    def on_train_epoch_end(self) -> None:
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        return self.optimizer


@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default="/home/matt/PycharmProjects/ULES/config/config.yaml")
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
@click.option('--double_backbone/--single_backbone',
              show_default=True,
              help='if triggered, use double backbone',
              default=True)
@click.option('--dataset_name',
              '-dtst',
              type=str,
              help='name of the dataset to be pre-trained.',
              default=None)
def main(config, extra, checkpoint, batch_size, num_workers, gpus, double_backbone, dataset_name):
    cfg = yaml.safe_load(open(config))  # load the config file

    # dataset_name = "Roses"
    # check if dataset name is available in the config
    if dataset_name in cfg["data"].keys():
        image_size = cfg["data"][dataset_name]['image_size']
        mean = cfg["data"][dataset_name]['mean']  # currently not implemented to change mean and std from here
        std = cfg["data"][dataset_name]['std']
        print("Starting pre-training for the %s dataset" % dataset_name)
    else:
        raise Exception("No dataset named {}".format(dataset_name))

    # check for the checkpoint save path to be available
    datapath_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', dataset_name).lower()
    version_name = (datapath_snake + "_db_final_l3") if double_backbone else (datapath_snake + "_sb_final_l3")
    checkpoint_save_path = join(join("checkpoints", datapath_snake), "pixpro_" + version_name + ".ckpt")
    if os.path.isfile(checkpoint_save_path):
        answer = input("We detected a checkpoint already saved with the same name. \n Do you want to overwrite? [Y/N]", )
        if answer.upper() in ["Y", "YES"]:
            print("Existing checkpoint will be overwritten.")
        else:
            raise Exception("Checkpoint name already existing. Please change the location.")

    # initialize the network to be fed to the PixPro arch
        resnet = models.segmentation.fcn_resnet50(pretrained=True, progress=True,
                                                  # num_classes=cfg["data"][dataset_name]['num_classes'], aux_loss=None)
                                                  aux_loss=None)
    resnet.classifier[-1] = torch.nn.Conv2d(512, cfg["data"][dataset_name]['num_classes'], kernel_size=(1, 1),
                                            stride=(1, 1))

    # checkpoint = "checkpoints/kyoto_material_seg/pixpro_kyoto_material_seg_db_final2.ckpt"  # hard coded checkpoint for training resume

    # initialize learner and optimizer to pass to the Lightning module
    # double_backbone = True
    # TODO: implement a flag to switch between the two options with the same file
    #  instead of this ugly if-else block
    if double_backbone:
        learner = PixelCL_DB(
            resnet.backbone,  # only pass the backbone of the FCN ResNet
            image_size=image_size,
            hidden_layer_pixel='layer3',  # leads to output of feature map for pixel-level learning
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
            use_pixpro=True,
            # do pixel pro instead of pixel contrast loss, defaults to pixpro, since it is the best one
            cutout_ratio_range=(0.6, 0.8),  # a random ratio is selected from this range for the random cutout
            use_range_image=True
        )
        print("Running a double backbone pre-training.")
    else:
        learner = PixelCL(
            resnet.backbone,  # only pass the backbone of the FCN ResNet
            image_size=image_size,
            hidden_layer_pixel='layer3',  # leads to output of feature map for pixel-level learning
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
            use_pixpro=True,
            # do pixel pro instead of pixel contrast loss, defaults to pixpro, since it is the best one
            cutout_ratio_range=(0.6, 0.8),  # a random ratio is selected from this range for the random cutout
            use_range_image=True
        )
        print("Running a single backbone pre-training.")

    opt = torch.optim.AdamW(learner.parameters())
    # opt = torch.optim.SGD(learner.parameters(), lr=1e-2, momentum=0.9)

    model = LightningPixelCL(learner, opt)

    checkpoint_callback = ModelCheckpoint(dirpath=join("checkpoints", datapath_snake), save_top_k=2, monitor="loss",
                                          every_n_epochs=1, filename="{epoch}-%s" % version_name)
    torch.set_float32_matmul_precision('medium')

    tb_logger = TensorBoardLogger(save_dir=os.getcwd(), name="unsup_logs", version=version_name)

    trainer = pl.Trainer(devices=gpus, logger=tb_logger, callbacks=[checkpoint_callback, StochasticWeightAveraging(swa_lrs=1e-2)],
                         max_epochs=150, accumulate_grad_batches=8, precision="bf16-mixed", # detect_anomaly=True,
                         gradient_clip_algorithm="norm")

    # initialize the datasets
    if extra:  # this flag is only necessary for the Cityscapes dataset
        split_train = 'train_extra'
        mode = 'coarse'
    else:
        split_train = 'train'
        mode = 'fine'

    match dataset_name:
        # potentially obsolete
        # case "Cityscapes":
        #     train_data = CityDataPixel(city_cfg["data"][dataset_name]['location'], split=split_train, mode=mode, target_type='semantic', transforms=None)
        case "KittiSem":
            train_data = KittiRangeDataset(root_dir=cfg["data"][dataset_name]['location'], split="train",
                                           image_size=image_size)
        case "KyotoMaterialSeg":
            train_data = MultimodalMaterialDataset(root_dir=cfg["data"][dataset_name]['location'], split="train",
                                                   image_size=image_size)
        case "VisNir":
            data = VisNirDataset(root_dir=cfg["data"][dataset_name]['location'], image_size=image_size)
        case "Roses":
            data = MultiSpectralCropWeed(root_dir=cfg["data"][dataset_name]['location'], image_size=image_size,
                                         split="roses23")

    if dataset_name == "Roses" or dataset_name == "VisNir":
        # prepare the dataset split for the datasets that require it

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

    batch_size = batch_size

    trainer.fit(model,
                DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                           drop_last=True),
                ckpt_path=checkpoint)

    trainer.save_checkpoint(checkpoint_save_path)


if __name__ == '__main__':
    main()
