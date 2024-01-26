from os.path import join, dirname, abspath
import yaml
from models import ULES, ULES_DB
import torch
from torchvision import models
from unsup_pretrain.pixel_level_contrastive_learning.pixel_level_contrastive_learning_DB import PixelCL_DB
# from unsup_pretrain.pixelpro import LightningPixelCL


def dictionary_parse_to_ules(ckpt_state_dict, rgb_only_ft: bool, double_backbone: bool, only_bb: bool):
    loopable_state_dict = dict(ckpt_state_dict)

    # following loop is to adapt from pixel pro learner weights
    for k in loopable_state_dict.keys():
        if k.startswith("backbone.0."):
            new_k = k.replace("backbone.0.", "backbone.conv1.")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
        if k.startswith("backbone.1."):
            new_k = k.replace("backbone.1.", "backbone.bn1.")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
    loopable_state_dict = dict(ckpt_state_dict)

    # remove an unused key
    for k in loopable_state_dict.keys():
        if k.endswith("num_batches_tracked"):
            del ckpt_state_dict[k]
    loopable_state_dict = dict(ckpt_state_dict)

    # all these following loops are to adapt the key names to our architecture
    for k in loopable_state_dict.keys():
        k_str = str(k)
        k_split = k_str.split(sep=".")
        if len(k_split) == 3:
            continue
        k_suffix = k_split[1]
        if k.startswith("backbone." + k_suffix + "."):
            k_suffix_new = str(int(k_suffix) - 3)
            new_k = k.replace("backbone." + k_suffix + ".", "backbone.layer" + k_suffix_new + ".")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
    loopable_state_dict = dict(ckpt_state_dict)

    for k in loopable_state_dict.keys():
        if k.startswith("backbone"):
            new_k = k.replace("backbone.", "model.backbone.")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
    loopable_state_dict = dict(ckpt_state_dict)

    for k in loopable_state_dict.keys():
        if k.startswith("model.online_encoder.net"):
            new_k = k.replace("model.online_encoder.net.", "model.")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
    loopable_state_dict = dict(ckpt_state_dict)

    for k in loopable_state_dict.keys():
        if k.startswith("model.online_encoder_rgb"):
            new_k = k.replace("model.online_encoder_rgb.net.backbone.", "model.backbone_rgb.")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
    loopable_state_dict = dict(ckpt_state_dict)

    # in the pretraining arch I decided to call the range view backbone "gray" because it was a grayscale image
    # in the end they are brought back to a specific color space anyway, so I decided to use "range" here
    for k in loopable_state_dict.keys():
        if k.startswith("model.online_encoder_gray"):
            new_k = k.replace("model.online_encoder_gray.net.backbone.", "model.backbone_range.")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
    loopable_state_dict = dict(ckpt_state_dict)

    # this block is to only use the rgb weights of the pretrained double backbone
    if rgb_only_ft and not double_backbone:
        for k in loopable_state_dict.keys():
            if k.startswith("model.backbone_rgb"):
                new_k = k.replace("model.backbone_rgb.", "model.backbone.")
                ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
        loopable_state_dict = dict(ckpt_state_dict)

    # remove everything that isn't backbone
    if only_bb:
        for k in loopable_state_dict.keys():
            # if not k.startswith("model.backbone"):
            if not "backbone" in k:
                del ckpt_state_dict[k]
    loopable_state_dict = dict(ckpt_state_dict)

    # remove the first layer if we're only using the image for image+range pretraining
    # if range_pt:
    #     for k in loopable_state_dict.keys():
    #         if k.startswith("model.backbone.conv1"):
    #             del state_dict[k]

    return ckpt_state_dict


def dictionary_parse_to_pixpro(ckpt_state_dict, rgb_only_ft: bool, double_backbone: bool, only_bb: bool):
    loopable_state_dict = dict(ckpt_state_dict)

    # following loop is to adapt from pixel pro learner weights
    for k in loopable_state_dict.keys():
        if k.startswith("backbone.0."):
            new_k = k.replace("backbone.0.", "backbone.conv1.")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
        if k.startswith("backbone.1."):
            new_k = k.replace("backbone.1.", "backbone.bn1.")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
    loopable_state_dict = dict(ckpt_state_dict)

    # remove an unused key
    for k in loopable_state_dict.keys():
        if k.endswith("num_batches_tracked"):
            del ckpt_state_dict[k]
    loopable_state_dict = dict(ckpt_state_dict)

    # all these following loops are to adapt the key names to our architecture
    for k in loopable_state_dict.keys():
        k_str = str(k)
        k_split = k_str.split(sep=".")
        if len(k_split) == 3:
            continue
        k_suffix = k_split[1]
        if k.startswith("backbone." + k_suffix + "."):
            k_suffix_new = str(int(k_suffix) - 3)
            new_k = k.replace("backbone." + k_suffix + ".", "backbone.layer" + k_suffix_new + ".")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
    loopable_state_dict = dict(ckpt_state_dict)

    for k in loopable_state_dict.keys():
        if k.startswith("backbone"):
            new_k = k.replace("backbone.", "model.backbone.")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
    loopable_state_dict = dict(ckpt_state_dict)

    for k in loopable_state_dict.keys():
        if k.startswith("model.online_encoder.net"):
            new_k = k.replace("model.online_encoder.net.", "model.")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
    loopable_state_dict = dict(ckpt_state_dict)

    for k in loopable_state_dict.keys():
        if k.startswith("model.online_encoder_rgb"):
            new_k = k.replace("model.online_encoder_rgb.net.backbone.", "model.backbone_rgb.")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
    loopable_state_dict = dict(ckpt_state_dict)

    # in the pretraining arch I decided to call the range view backbone "gray" because it was a grayscale image
    # in the end they are brought back to a specific color space anyway, so I decided to use "range" here
    for k in loopable_state_dict.keys():
        if k.startswith("model.online_encoder_gray"):
            new_k = k.replace("model.online_encoder_gray.net.backbone.", "model.backbone_range.")
            ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
    loopable_state_dict = dict(ckpt_state_dict)

    # this block is to only use the rgb weights of the pretrained double backbone
    if rgb_only_ft and not double_backbone:
        for k in loopable_state_dict.keys():
            if k.startswith("model.backbone_rgb"):
                new_k = k.replace("model.backbone_rgb.", "model.backbone.")
                ckpt_state_dict[new_k] = ckpt_state_dict.pop(k)
        loopable_state_dict = dict(ckpt_state_dict)

    # remove everything that isn't backbone
    if only_bb:
        for k in loopable_state_dict.keys():
            # if not k.startswith("model.backbone"):
            if not "backbone" in k:
                del ckpt_state_dict[k]
    loopable_state_dict = dict(ckpt_state_dict)

    return ckpt_state_dict


if __name__ == "__main__":
    config = join(dirname(abspath(__file__)), 'config/config.yaml')
    cfg = yaml.safe_load(open(config))

    rgb_only_ft = True
    double_backbone = False
    only_bb = True

    resnet = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=20,
                                              aux_loss=None)
    learner = PixelCL_DB(
        resnet,
        image_size=[90, 160],
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
        cutout_ratio_range=(0.6, 0.8),  # a random ratio is selected from this range for the random cutout
        use_range_image=True
    )

    opt = torch.optim.AdamW(learner.parameters(), lr=1e-4)

    # model = LightningPixelCL(learner, opt)
    # checkpoint = torch.load("checkpoints/db_ft_100%.ckpt")
    # state_dict = checkpoint["state_dict"]
    # state_dict = dictionary_parse_to_ules(state_dict)
    # model.load_state_dict(state_dict, strict=False)
    # odd_keys = model.load_state_dict(state_dict, strict=False)
