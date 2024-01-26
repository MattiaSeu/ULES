import PIL.Image
import click
import os
from os.path import join, dirname, abspath
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.transforms import ToTensor, Normalize, Compose
import yaml
from datasets.datasets import StatDataModule
from models.ULES import Ules as ULES
from models.ULES_DB import Ules as ULES_DB
import torch.nn.functional as F
from utils.tenprint import print_tensor
from utils.segmap import kitti_decode
from utils.state_dictionary_helpers import dictionary_parse_to_ules
import matplotlib.pyplot as plt

config = join(dirname(abspath(__file__)), 'config/config.yaml')
cfg = yaml.safe_load(open(config))
model_full = ULES_DB.load_from_checkpoint("checkpoints/db_ft_100%.ckpt", cfg=cfg, rgb_only_ft=False)
model_rgb = ULES_DB.load_from_checkpoint("checkpoints/db_ft_100%.ckpt", cfg=cfg, rgb_only_ft=True)

rgb_only_ft = True
double_backbone = False
only_bb = False
model2 = ULES(cfg=cfg)
model1 = ULES(cfg=cfg)
checkpoint = torch.load("checkpoints/sb_ft_100%.ckpt")
state_dict = checkpoint["state_dict"]
state_dict = dictionary_parse_to_ules(state_dict, rgb_only_ft, double_backbone, only_bb)
model1.load_state_dict(state_dict, strict=False)
odd_keys = model1.load_state_dict(state_dict, strict=False)

if len(odd_keys[0]) > 1:
    print("The following keys are missing:")
    for miss_key in odd_keys[0]:
        print(miss_key)
    if odd_keys[1]:
        print("The following keys are unexpected:")
        for unexpected_key in odd_keys[1]:
            print(unexpected_key)

transform = Compose([
    ToTensor(),
    # Normalize(mean=(0.288, 0.296, 0.299), std=(0.285, 0.299, 0.311)),

]
)

rootdir = "/home/matt/data/kitti_sem/test/rgb/"
# rangedir = "/home/matt/data/kitti_sem/train/gray/"
counter = 0
rangedir = join("/home/matt/data/kitti_sem/test/", "sequences")

for image_name in os.listdir(rootdir):
    seq_name_list = image_name.split(sep=".")[0].split(sep="_")
    range_name = seq_name_list[0] + "/range_projection/" + seq_name_list[1] + ".png"
    image_og = PIL.Image.open(join(rootdir, image_name))
    range_og = PIL.Image.open(join(rangedir, range_name)).convert("LA")
    image_size = image_og.size
    # for i in range(6):
        # image_res = image_og.resize((image_size[0] // (i + 1), image_size[1] // (i + 1)))
        # range_res = range_og.resize((image_size[0] // (i + 1), image_size[1] // (i + 1)))
    image_res = image_og.resize((160, 90))
    range_res = range_og.resize((160, 90))
    image = transform(image_res)
    range_view = ToTensor()(range_res)
    range_view = range_view[1]
    image = image.unsqueeze(dim=0)
    range_view = range_view.unsqueeze(dim=0)
    range_view = range_view.unsqueeze(dim=0)
    with (torch.no_grad()):
        input_shape = image.shape[-2:]

        out_full = model_full({"image": image, "range_view": range_view})
        out_full = torch.argmax(torch.softmax(out_full[0], 0), 0)
        decoded_full = kitti_decode(out_full)

        out = model_rgb.model.backbone_rgb(image)
        out = model_rgb.model.head(out["out"])
        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
        out = torch.argmax(torch.softmax(out[0], 0), 0)
        decoded_out = kitti_decode(out)

        out_sb = model1(image)
        out_sb = torch.argmax(torch.softmax(out_sb[0], 0), 0)
        decoded_sb = kitti_decode(out_sb)

        out_sb_empty = model2(image)
        out_sb_empty = torch.argmax(torch.softmax(out_sb_empty[0], 0), 0)
        decoded_sb_empty = kitti_decode(out_sb_empty)

        ypixels = image.shape[-2]
        xpixels = image.shape[-1]
        dpi = 96.
        xinch = xpixels / dpi
        yinch = ypixels / dpi
        plt.figure(figsize=(xinch * 4, yinch * 1.5), dpi=96)
        fig, ax = plt.subplots(ncols=2, nrows=2)
        ax[0][0].imshow(image_res)
        ax[0][1].imshow(decoded_out)
        ax[1][0].imshow(decoded_full)
        ax[1][1].imshow(decoded_sb)
        ax[0][0].axis('off')
        ax[0][1].axis('off')
        ax[1][0].axis('off')
        ax[1][1].axis('off')
        ax[0][0].set_title('Input Image')
        ax[0][1].set_title('Db rgb')
        ax[1][0].set_title('Db full')
        ax[1][1].set_title('Sb rgb')
        plt.tight_layout()
        plt.margins(x=0)
        plt.margins(y=0)
        # plt.savefig(join(os.getcwd(), f"output/kitti/OG/{image_name}"), bbox_inches = "tight", dpi=200)
        plt.show()
        # plt.close()
    if counter == 100:
        break
    else:
        counter += 1
