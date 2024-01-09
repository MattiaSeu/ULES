import copy
import random
from typing import Any, Tuple

import torch
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from torchvision import transforms
from torchvision.datasets import Cityscapes

from transforms_custom import RandomFlipWithReturn, RandomSizeCropWithCoord


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


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
                image_contr, x_crop, y_crop, crop_width, crop_height = transform(image_contr)
                if x_crop != -1:
                    target_t = target_t.crop([x_crop, y_crop, x_crop + crop_width, y_crop + crop_height])
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


class CityDataPixel(Cityscapes):

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

        return transform_image(image)
