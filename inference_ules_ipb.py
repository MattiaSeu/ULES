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
model = ULES_DB.load_from_checkpoint("checkpoints/db_ft_100%.ckpt", cfg=cfg, rgb_only_ft=True)
model.eval()

# model1 = ULES_DB.load_from_checkpoint("checkpoints/db_ft_rgb_100%.ckpt", cfg=cfg, rgb_only_ft=True)
# model1.eval()

rgb_only_ft = True
double_backbone = False
only_bb = True
model1 = ULES(cfg=cfg)
checkpoint = torch.load("checkpoints/sb_ft_100%.ckpt")
state_dict = checkpoint["state_dict"]
state_dict = dictionary_parse_to_ules(state_dict)
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
    Normalize(mean=(0.288, 0.296, 0.299), std=(0.285, 0.299, 0.311)),

    ]
)

count = 0
rootdir = "/home/matt/data/ipb_car/2019-11-25_second_recording/images/"
for image_name in os.listdir(rootdir):
    if count == 100:
        break
    image_og = PIL.Image.open(join(rootdir, image_name))
    image_size = image_og.size
    for i in range(6):
        image_res = image_og.resize((image_size[0]//(i+1), image_size[1]//(i+1)))
        # image_og = image.resize((160, 90))
        image = transform(image_res)
        image = image.unsqueeze(dim=0)
        with (torch.no_grad()):
            input_shape = image.shape[-2:]
            out = model.model.backbone_rgb(image)
            out = model.model.head(out["out"])
            out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
            out = torch.argmax(torch.softmax(out[0], 0), 0)
            decoded_out = kitti_decode(out)

            # out1 = model1.model.backbone_rgb(image)
            # out1 = model1.model.head(out1["out"])
            # out1 = F.interpolate(out1, size=input_shape, mode="bilinear", align_corners=False)
            # out1 = torch.argmax(torch.softmax(out1[0], 0), 0)
            # decoded_out1 = kitti_decode(out1)

            out1 = model1(image)
            out1 = torch.argmax(torch.softmax(out1[0], 0), 0)
            decoded_out1 = kitti_decode(out1)

            ypixels = image.shape[-2]
            xpixels = image.shape[-1]
            dpi = 96.
            xinch = xpixels / dpi
            yinch = ypixels / dpi
            plt.figure(figsize=(xinch*4,yinch*1.5), dpi=96)
            fig, ax = plt.subplots(ncols=3)
            ax[0].imshow(image_res)
            ax[1].imshow(decoded_out)
            ax[2].imshow(decoded_out1)
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            ax[0].set_title('Input Image')
            ax[1].set_title('Db predict')
            ax[2].set_title('Sb predict')
            plt.tight_layout()
            plt.margins(x=0)
            plt.margins(y=0)
            plt.savefig(join(os.getcwd(), f"output/{str(i)}/{image_name}"), bbox_inches = "tight", dpi=200)
            # plt.show()
            plt.close()
        count += 1

