import copy

import PIL.ImageOps
from torchvision.datasets import Cityscapes
import os
import random
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageFilter
from torchvision import transforms
from transforms_custom import RandomCropWithCoord, RandomFlipWithReturn, RandomSizeCropWithCoord
from VicREGL_transforms import MultiCropTrainDataTransform
from PIL import ImageOps

from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets import Cityscapes

city_data_path = os.path.join("~/data", 'cityscapes/')
dataset = Cityscapes(city_data_path, split='train', mode='fine',
                     target_type='semantic')

ignore_index = 255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [ignore_index, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
               'train', 'motorcycle', 'bicycle']

class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes = len(valid_classes)

colors = [[0, 0, 0],
          [128, 64, 128],
          [244, 35, 232],
          [70, 70, 70],
          [102, 102, 156],
          [190, 153, 153],
          [153, 153, 153],
          [250, 170, 30],
          [220, 220, 0],
          [107, 142, 35],
          [152, 251, 152],
          [0, 130, 180],
          [220, 20, 60],
          [255, 0, 0],
          [0, 0, 142],
          [0, 0, 70],
          [0, 60, 100],
          [0, 80, 100],
          [0, 0, 230],
          [119, 11, 32],
          ]

label_colours = dict(zip(range(n_classes), colors))


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def encode_segmap(mask):
    # remove unwanted classes and rectify the labels of wanted classes
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask


def decode_segmap(temp):
    # convert gray scale to color
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


# old version of the dataloader
class CityDataContrastive(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        # transformations to apply

        transform_image = transforms.Compose(
            [
                transforms.Resize((128, 256), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.28689554, 0.32513303, 0.28389177),
                #                      std=(0.18696375, 0.19017339, 0.18720214)),
            ]
        )

        transform_contrastive = [
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            RandomFlipWithReturn(),
            RandomSizeCropWithCoord(),
            transforms.Resize((128, 256), transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.28689554, 0.32513303, 0.28389177),
            #                      std=(0.18696375, 0.19017339, 0.18720214)),
        ]

        transform_target = transforms.Compose(
            [
                transforms.Resize((128, 256), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ]
        )

        # initialize a deep copy for the view and the transformed mask
        image_contr = copy.deepcopy(image)
        target_t = copy.deepcopy(target)
        for num, transform in enumerate(transform_contrastive):
            if num not in [3, 4]:
                image_contr = transform(image_contr)  # simply apply transforms for all but our custom transforms
            elif num == 3:
                image_contr, flip_flag = transform(image_contr)
                if flip_flag:
                    target_t = ImageOps.mirror(target)
            else:
                image_contr, x_crop, y_crop = transform(image_contr)
                if x_crop != -1:
                    target_t = target_t.crop([x_crop, y_crop, x_crop+256, y_crop+128])
        target_t = transform_target(target_t)

        data_out = {'image': transform_image(image), 'img_contrastive': image_contr,
                    'target': (transform_target(target).squeeze(0) * 255).type(torch.int64),
                    'target_t': (target_t.squeeze(0) * 255).type(torch.int64),
                    'flip_flag': flip_flag, 'crop_pos': [x_crop, y_crop]}

        # return transformed['image'], transformed['mask']

        if len(data_out['image'].shape) != 3:
            data_out['image'] = data_out['image'].unsqueeze(0)

        # data_out['image'] = data_out['image'].permute(1, 2, 0)
        # data_out['mask'] = data_out['mask'].permute(1, 2, 0).squeeze()
        return data_out
    # torch.unsqueeze(transformed['mask'], 0)


class CityData(Cityscapes):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        # transformations to apply

        transform_image = transforms.Compose(
            [
                transforms.Resize((128, 256), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor()
            ]
        )

        transform_target = transforms.Compose(
            [
                transforms.Resize((128, 256), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ]
        )

        tra = {'image': transform_target(image),
               'target': (transform_target(target).squeeze(0) * 255).type(torch.int64)}

        # return transformed['image'], transformed['mask']

        if len(tra['image'].shape) != 3:
            tra['image'] = tra['image'].unsqueeze(0)

        # tra['image'] = tra['image'].permute(1, 2, 0)
        # tra['mask'] = tra['mask'].permute(1, 2, 0).squeeze()
        return tra['image'], tra['target']
    # torch.unsqueeze(transformed['mask'], 0)
