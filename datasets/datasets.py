import math

import torch
import yaml
import torchvision
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import skimage.transform
from torchvision.datasets import Cityscapes
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset
from pytorch_lightning import LightningDataModule
import os
import os.path as path
from PIL import Image, ImageFile
# import utils.utils as utils
import numpy as np
import torch.nn.functional as F
import glob
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class StatDataModule(LightningDataModule):
    def __init__(self, cfg, reduced_data):
        super().__init__()
        # from cfg I can access all my stuff
        # as data path, data size and so on 
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.cfg = cfg
        self.data_ratio = reduced_data
        self.len = -1
        self.setup()
        self.loader = [self.train_dataloader(), self.val_dataloader(), self.test_dataloader(),
                       self.predict_dataloader()]

    def prepare_data(self):
        # Augmentations are applied using self.transform 
        # no data to download, for now everything is local 
        pass

    def setup(self, stage=None):

        self.mode = self.cfg['train']['mode']

        if "cityscape" in self.cfg['data']['ft-path']:
            if stage == 'fit' or stage is None:
                self.data_train = CityData(self.cfg['data']['ft-path'], split='train',
                                           mode='fine',
                                           target_type='semantic')
                self.data_val = CityData(self.cfg['data']['ft-path'], split='val',
                                         mode='fine',
                                         target_type='semantic')
                self.data_test = CityData(self.cfg['data']['ft-path'], split='val',
                                          mode='fine',
                                          target_type='semantic')
        elif "kitti" in self.cfg['data']['ft-path']:
            if stage == 'fit' or stage is None:
                self.data_train = KittiRangeDataset_DB(self.cfg['data']['ft-path'], split='train')
                self.data_val = KittiRangeDataset_DB(self.cfg['data']['ft-path'], split='test')
        elif "ipb" in self.cfg['data']['ft-path']:
            if stage == 'fit' or stage is None:
                self.data_train = IPB_Car(self.cfg['data']['ft-path'])
                self.data_val = IPB_Car(self.cfg['data']['ft-path'])
        elif "multi" in self.cfg['data']['ft-path']:
            if stage == 'fit' or stage is None:
                self.data_train = MultimodalMaterial(self.cfg['data']['ft-path'], split='train', image_size=[128, 153])
                self.data_val = MultimodalMaterial(self.cfg['data']['ft-path'], split='test', image_size=[128, 153])
        return

    def train_dataloader(self):
        if self.mode == 'eval': pass
        if self.data_ratio != 100:
            subset_len_train = round(len(self.data_train) * (self.data_ratio / 100))
            self.data_train = Subset(self.data_train, indices=range(0, subset_len_train))

        loader = DataLoader(self.data_train,
                            batch_size=self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            num_workers=self.cfg['train']['workers'],
                            pin_memory=True,
                            shuffle=True)
        self.len = self.data_train.__len__()
        return loader

    def val_dataloader(self):
        if self.mode == 'pt':
            pass
        elif self.mode == 'eval':
            pass
        loader = DataLoader(self.data_val,
                            self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            num_workers=self.cfg['train']['workers'],
                            pin_memory=True,
                            shuffle=False)
        self.len = self.data_val.__len__()
        return loader

    def test_dataloader(self):
        if self.mode != "eval": return
        loader = DataLoader(self.data_test,
                            self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            num_workers=self.cfg['train']['workers'],
                            pin_memory=True,
                            shuffle=False)
        self.len = self.data_test.__len__()
        return loader

    def predict_dataloader(self):
        if self.mode != "infer": return
        loader = DataLoader(self.data_train,
                            self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            num_workers=self.cfg['train']['workers'],
                            pin_memory=True,
                            shuffle=False)
        self.len = self.data_train.__len__()
        return loader


#################################################
#################### Datasets ###################
#################################################

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

        tra = {'image': transform_image(image), 'target': (transform_target(target).squeeze(0) * 255).type(torch.int64)}

        # return transformed['image'], transformed['mask']

        if len(tra['image'].shape) != 3:
            tra['image'] = tra['image'].unsqueeze(0)

        # tra['image'] = tra['image'].permute(1, 2, 0)
        # tra['mask'] = tra['mask'].permute(1, 2, 0).squeeze()
        return tra
    # torch.unsqueeze(transformed['mask'], 0)


import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class KittiRangeDataset(Dataset):

    def __init__(self, root_dir, split: str, transform=None):
        """
        Arguments:.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.transform = transform
        self.image_list = os.listdir(self.root_dir + "/rgb/")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb_root = os.path.join(self.root_dir, "rgb")
        target_root = os.path.join(self.root_dir, "gray")
        range_root = os.path.join(self.root_dir, "sequences")

        img_path = os.path.join(rgb_root, self.image_list[idx])

        image = Image.open(img_path).convert('RGB')

        seq_name_list = self.image_list[idx].split(sep=".")[0].split(sep="_")

        target_path = os.path.join(target_root, f"{seq_name_list[0]}_{seq_name_list[1]}.png")

        target = Image.open(target_path).convert('L')

        range_view = Image.open(range_root + "/" + seq_name_list[0] + "/range_projection/" +
                                seq_name_list[1] + ".png").convert("LA")

        transform_image = transforms.Compose(
            [
                transforms.Resize((90, 160), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

        transform_range = transforms.Compose(
            [
                transforms.Resize((90, 160), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

        image = transform_image(image)
        target = transform_range(target).squeeze(0) * 255
        range_view = transform_image(range_view)

        range_view = range_view[1]
        range_view = range_view[None, :, :]
        sample = {"image": torch.cat((image, range_view), 0), "target": target.type(torch.int64)}

        return sample


class IPB_Car(Dataset):

    def __init__(self, root_dir: str, transform=None):
        """
        Arguments:.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = os.path.join(root_dir)
        self.transform = transform
        self.image_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.image_list[idx])

        image = Image.open(img_path).convert('RGB')

        transform_image = transforms.Compose(
            [
                transforms.Resize((90, 160), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

        image = transform_image(image)

        sample = {"image": image}

        return sample


class KittiRangeDataset_DB(Dataset):

    def __init__(self, root_dir, split: str, transform=None):
        """
        Arguments:.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.transform = transform
        self.image_list = os.listdir(self.root_dir + "/gray/")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb_root = os.path.join(self.root_dir, "rgb")
        target_root = os.path.join(self.root_dir, "gray")
        range_root = os.path.join(self.root_dir, "sequences")

        img_path = os.path.join(rgb_root, self.image_list[idx])

        image = Image.open(img_path).convert('RGB')

        seq_name_list = self.image_list[idx].split(sep=".")[0].split(sep="_")

        target_path = os.path.join(target_root, f"{seq_name_list[0]}_{seq_name_list[1]}.png")

        target = Image.open(target_path).convert('L')

        range_view = Image.open(range_root + "/" + seq_name_list[0] + "/range_projection/" +
                                seq_name_list[1] + ".png").convert("LA")

        og_size = image.size
        # new_size = tuple(math.floor(s/4) for s in og_size)
        new_size = (90, 160)
        kitti_mean = (0.35095342, 0.36734804, 0.36330285)
        kitti_std = (0.30601038, 0.31168418, 0.32000023)

        transform_image = transforms.Compose(
            [
                transforms.Resize(new_size, transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=kitti_mean, std=kitti_std)
            ]
        )

        transform_range = transforms.Compose(
            [
                transforms.Resize(new_size, transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor()
            ]
        )

        transform_range = transforms.Compose(
            [
                transforms.Resize(new_size, transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

        image = transform_image(image)
        target = transform_range(target).squeeze(0) * 255
        range_view = transform_range(range_view)

        range_view = range_view[1]
        range_view = range_view[None, :, :]
        sample = {"image": image, "range_view": range_view, "target": target.type(torch.int64)}

        return sample


class MultimodalMaterial(Dataset):

    def __init__(self, root_dir, image_size, split, transform=None):
        """
        Arguments:.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        list_dir = os.path.join(root_dir, 'list_folder')
        with open(os.path.join(os.path.join(list_dir, split + '.txt')), "r") as f:
            self.image_list = f.read().splitlines()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        color_dir = os.path.join(self.root_dir, 'polL_color')
        aolp_sin_dir = os.path.join(self.root_dir, 'polL_aolp_sin')
        aolp_cos_dir = os.path.join(self.root_dir, 'polL_aolp_cos')
        dolp_dir = os.path.join(self.root_dir, 'polL_dolp')
        gt_dir = os.path.join(self.root_dir, 'GT')

        color_name = os.path.join(color_dir, self.image_list[idx] + ".png")
        color_image = Image.open(color_name).convert('RGB')
        aolp_sin_name = os.path.join(aolp_sin_dir, self.image_list[idx] + ".npy")
        aolp_sin = np.load(aolp_sin_name)
        aolp_cos_name = os.path.join(aolp_cos_dir, self.image_list[idx] + ".npy")
        aolp_cos = np.load(aolp_cos_name)
        aolp = np.stack([aolp_sin, aolp_cos], axis=2)
        gt_name = os.path.join(gt_dir, self.image_list[idx] + ".png")
        gt = Image.open(gt_name).convert('L')

        dolp_name = os.path.join(dolp_dir, self.image_list[idx] + ".npy")
        dolp = np.load(dolp_name)

        daolp = np.stack([aolp_sin, aolp_cos, dolp], axis=2)
        daolp = skimage.transform.resize(daolp, (self.image_size[0], self.image_size[1]), order=1, mode="edge")
        dolp = skimage.transform.resize(dolp, (self.image_size[0], self.image_size[1]), order=1, mode="edge")

        transform_image = transforms.Compose(
            [
                transforms.Resize((self.image_size[0], self.image_size[1]), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        transform_pol = transforms.Compose(
            [
                # transforms.Resize((self.image_size[0], self.image_size[1]), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        transform_gt = transforms.Compose(
            [
                transforms.Resize((self.image_size[0], self.image_size[1]), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

        color_image = transform_image(color_image)
        daolp = transform_pol(daolp)
        dolp = transform_pol(dolp)
        gt = transform_gt(gt).squeeze(0) * 255

        # sample = {'image': color_image, 'aolp': aolp, 'dolp': dolp}
        sample = {'image': color_image, 'range_view': dolp, "target": gt}

        # range_view = transform_image(range_view)
        # range_view = range_view[1]
        # range_view = range_view[None, :, :]
        # sample = torch.cat((color_image, range_view), 0)

        return sample
