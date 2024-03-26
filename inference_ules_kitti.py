import PIL.Image
from skimage.segmentation import slic
import numpy as np
import click
import os
from os.path import join, dirname, abspath
import torch
from torchvision.transforms import ToTensor, Normalize, Compose
import yaml
from models.ULES import Ules as ULES
from models.ULES_DB import Ules as ULES_DB
import torch.nn.functional as F
from utils.tenprint import print_tensor
from utils.segmap import kitti_decode
from utils.state_dictionary_helpers import dictionary_parse_to_ules
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def segment_to_onehot(superpix_seg):
    one_hot_masks = []
    mask_values_list = []

    mask_values = np.unique(superpix_seg)
    mask_list = []

    for label in mask_values:
        mask = np.zeros_like(superpix_seg)
        mask[superpix_seg == label] = 1
        mask_list.append(mask)

    one_hot_masks.append(mask_list)
    mask_values_list.append(mask_values)

    return one_hot_masks, mask_values_list


config = join(dirname(abspath(__file__)), 'config/config.yaml')
cfg = yaml.safe_load(open(config))
model_full = ULES_DB.load_from_checkpoint("checkpoints/db_ft_trial_100%.ckpt", cfg=cfg, rgb_only_ft=False)
model_rgb = ULES_DB.load_from_checkpoint("checkpoints/db_ft_trial_100%.ckpt", cfg=cfg, rgb_only_ft=True)

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
    Normalize(mean=(0.288, 0.296, 0.299), std=(0.285, 0.299, 0.311)),

]
)

rootdir = "/home/matt/data/kitti_sem/test/rgb/"
# rangedir = "/home/matt/data/kitti_sem/train/gray/"
counter = 0
rangedir = join("/home/matt/data/kitti_sem/test/", "sequences")

for image_name in os.listdir(rootdir):
    seq_name_list = image_name.split(sep=".")[0].split(sep="_")
    range_name = seq_name_list[0] + "/range_projection/" + seq_name_list[1] + ".png"
    label_name = "/home/matt/data/kitti_sem/test/labels/" + image_name
    image_og = PIL.Image.open(join(rootdir, image_name))
    range_og = PIL.Image.open(join(rangedir, range_name)).convert("LA")
    labels = PIL.Image.open(label_name)
    image_size = image_og.size
    # new_size = (310, 94)
    new_size = (160, 90)
    # for i in range(6):
    # image_res = image_og.resize((image_size[0] // (i + 1), image_size[1] // (i + 1)))
    # range_res = range_og.resize((image_size[0] // (i + 1), image_size[1] // (i + 1)))
    image_res = image_og.resize((new_size))
    range_res = range_og.resize((new_size))
    labels_res = labels.resize((new_size))
    image = transform(image_res)
    image_superpix = image.permute(1, 2, 0)
    image_superpix = image_superpix.numpy()
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

        numSegments = 100
        image_array = np.asarray(image_res).astype(np.float64)
        segments = slic(image_superpix, n_segments=numSegments, sigma=1, compactness=10, enforce_connectivity=True)
        one_hot_masks, mask_values_list = segment_to_onehot(segments)
        new_decode = decoded_full.copy()
        for mask in one_hot_masks[0]:
            broadcast_mask = mask[:, :, np.newaxis]
            masked_image = decoded_full * broadcast_mask
            values, counts = np.unique(masked_image.reshape(-1, masked_image.shape[2]), axis=0, return_counts=True)
            counts = counts[1:]  # remove 0 value from masking
            if len(counts) == 1:
                continue
            # plt.imshow(masked_image)
            # plt.show()
            count_tot = 0
            for count in counts:
                count_tot += count
            threshold = count_tot * 0.7
            if counts.max() > threshold:
                override_val = values[np.argmax(counts) + 1]
                new_decode[mask==1] = override_val


        ypixels = image.shape[-2]
        xpixels = image.shape[-1]
        dpi = 96.
        xinch = xpixels / dpi
        yinch = ypixels / dpi
        plt.figure(figsize=(xinch * 4, yinch * 1.5), dpi=96)
        fig, ax = plt.subplots(ncols=2, nrows=2)
        ax[0][0].imshow(image_res)
        ax[0][1].imshow(labels_res)
        ax[1][0].imshow(decoded_full)
        ax[1][1].imshow(decoded_sb)
        ax[0][0].axis('off')
        ax[0][1].axis('off')
        ax[1][0].axis('off')
        ax[1][1].axis('off')
        ax[0][0].set_title('Input Image')
        ax[0][1].set_title('Ground Truth')
        ax[1][0].set_title('DB pred')
        ax[1][1].set_title('SB pred')



        color_list = np.unique(decoded_full.reshape(-1, decoded_full.shape[2]), axis=0)
        kitti_colors_list = [
            [255, 0, 0],  # ignore
            [128, 0, 0],  # building
            [128, 64, 128],  # road
            [0, 0, 192],  # sidewalk
            [64, 64, 128],  # fence
            [128, 128, 0],  # vegetation
            [192, 192, 128],  # pole
            [64, 0, 128],  # car
            [192, 128, 128],  # sign
            [64, 64, 0],  # pedestrian
            [0, 128, 192],  # cyclist
            [128, 128, 128],  # sky
        ]
        kitti_colors_name_list = [
            "ignore",
            "building",
            "road",
            "sidewalk",
            "fence",
            "vegetation",
            "pole",
            "car",
            "sign",
            "pedestrian",
            "cyclist",
            "sky"
        ]
        kitti_colors_list = [[r / 255.0, g / 255.0, b / 255.0] for r, g, b in kitti_colors_list]
        legend_handles = [mpatches.Patch(color=kitti_colors_list[i], label=f'{kitti_colors_name_list[i]}') for i in range(len(kitti_colors_list))]
        ax[1][1].legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.6, 1))
        plt.tight_layout()
        plt.margins(x=0)
        plt.margins(y=0)
        # plt.show()
        plt.savefig(join(os.getcwd(), f"output/kitti/dice/{image_name}"), bbox_inches = "tight", dpi=200)
        plt.close()
    # if counter == 100:
    #     break
    # else:
    #     counter += 1
