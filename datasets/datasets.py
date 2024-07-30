import math

import torch
import yaml
import torchvision
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import skimage.transform
from skimage import io
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
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class StatDataModule(LightningDataModule):
    def __init__(self, cfg: dict, dataset_name: str,  reduced_data: int, image_size: Optional[List[int]],
                 mean: Optional[List[float]], std: Optional[List[float]]):
        super().__init__()
        # from cfg I can access all my stuff
        # as data path, data size and so on 
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.data_ratio = reduced_data
        self.image_size = image_size
        self.mean = mean
        self.std = std
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

        if "cityscape" in self.cfg['data'][self.dataset_name]['location']:
            if stage == 'fit' or stage is None:
                self.data_train = CityData(self.cfg['data'][self.dataset_name]['location'], split='train',
                                           mode='fine',
                                           target_type='semantic')
                self.data_val = CityData(self.cfg['data'][self.dataset_name]['location'], split='val',
                                         mode='fine',
                                         target_type='semantic')
                self.data_test = CityData(self.cfg['data'][self.dataset_name]['location'], split='val',
                                          mode='fine',
                                          target_type='semantic')
        elif "kitti" in self.cfg['data'][self.dataset_name]['location']:
            if stage == 'fit' or stage is None:
                self.data_train = KittiRangeDataset_DB(self.cfg['data'][self.dataset_name]['location'], split='train')
                self.data_val = KittiRangeDataset_DB(self.cfg['data'][self.dataset_name]['location'], split='test')
        elif "ipb" in self.cfg['data'][self.dataset_name]['location']:
            if stage == 'fit' or stage is None:
                self.data_train = IPB_Car(self.cfg['data'][self.dataset_name]['location'])
                self.data_val = IPB_Car(self.cfg['data'][self.dataset_name]['location'])
        elif "multi" in self.cfg['data'][self.dataset_name]['location']:
            if stage == 'fit' or stage is None:
                self.data_train = MultimodalMaterial(self.cfg['data'][self.dataset_name]['location'], split='train', image_size=self.image_size)
                self.data_val = MultimodalMaterial(self.cfg['data'][self.dataset_name]['location'], split='test', image_size=self.image_size)
        elif "vis-nir" in self.cfg['data'][self.dataset_name]['location']:
            if stage == 'fit' or stage is None:
                data = VisNirDataset(self.cfg['data'][self.dataset_name]['location'],  image_size=self.image_size)
                from torch.utils.data import DataLoader, Subset
                from sklearn.model_selection import train_test_split

                TEST_SIZE = 0.1
                SEED = 42

                # generate indices: instead of the actual data we pass in integers instead
                train_indices, test_indices = train_test_split(
                    range(len(data)),
                    test_size=TEST_SIZE,
                    random_state=SEED
                )

                # generate subset based on indices
                train_split = Subset(data, train_indices)
                val_split = Subset(data, test_indices)

                self.data_train = train_split
                self.data_val = val_split
        elif "roses" in self.cfg['data'][self.dataset_name]['location']:
            if stage == 'fit' or stage is None:
                data = MultiSpecCropDataset(self.cfg['data'][self.dataset_name]['location'], mean=self.mean, std=self.std,
                                            image_size=self.image_size, split="roses23")
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
                train_split = Subset(data, train_indices)
                val_split = Subset(data, test_indices)

                self.data_train = train_split
                self.data_val = val_split
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

        # store directory paths in var
        color_dir = os.path.join(self.root_dir, 'polL_color')
        aolp_sin_dir = os.path.join(self.root_dir, 'polL_aolp_sin')
        aolp_cos_dir = os.path.join(self.root_dir, 'polL_aolp_cos')
        dolp_dir = os.path.join(self.root_dir, 'polL_dolp')
        nir_dir = os.path.join(self.root_dir, 'NIR_warped')
        gt_dir = os.path.join(self.root_dir, 'GT')

        # load data
        color_name = os.path.join(color_dir, self.image_list[idx] + ".png")
        color_image = Image.open(color_name).convert('RGB')
        aolp_sin_name = os.path.join(aolp_sin_dir, self.image_list[idx] + ".npy")
        aolp_sin = np.load(aolp_sin_name)
        aolp_cos_name = os.path.join(aolp_cos_dir, self.image_list[idx] + ".npy")
        aolp_cos = np.load(aolp_cos_name)
        # aolp = np.stack([aolp_sin, aolp_cos], axis=2)

        nir_left_offset = 192
        nir_name = os.path.join(nir_dir, self.image_list[idx] + ".png")
        nir = Image.open(nir_name).convert('L')
        nir_cv2 = cv2.imread(nir_name, -1)
        # remove the nir camera offset
        nir_cv2 = skimage.transform.resize(nir_cv2[:, nir_left_offset:],
                                           (self.image_size[0], self.image_size[1]), order=1, mode="edge")
        gt_name = os.path.join(gt_dir, self.image_list[idx] + ".png")
        gt = Image.open(gt_name).convert('L')

        dolp_name = os.path.join(dolp_dir, self.image_list[idx] + ".npy")
        dolp = np.load(dolp_name)

        daolp = np.stack([aolp_sin, aolp_cos, dolp], axis=2)  # stack all reflection data in a single array
        nir_array = np.array(nir)
        daolpnir = np.stack([aolp_sin, aolp_cos, dolp, nir_array], axis=2)
        daolp = skimage.transform.resize(daolp, (self.image_size[0], self.image_size[1]), order=1, mode="edge")
        daolpnir = skimage.transform.resize(daolpnir, (self.image_size[0], self.image_size[1]), order=1, mode="edge")
        dolp = skimage.transform.resize(dolp, (self.image_size[0], self.image_size[1]), order=1, mode="edge")

        w, h = color_image.size
        color_image_offset = color_image.crop((nir_left_offset, 0, w, h))
        gt_offset = gt.crop((nir_left_offset, 0, w, h))

        transform_image = transforms.Compose(
            [
                transforms.Resize((self.image_size[0], self.image_size[1]), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
        transform_nir = transforms.Compose(
            [
                transforms.Resize((self.image_size[0], self.image_size[1]), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

        color_image = transform_image(color_image)
        color_image_offset = transform_image(color_image_offset)
        daolp = transform_pol(daolp)
        dolp = transform_pol(dolp)
        nir = transform_nir(nir)
        nir_cv2 = transform_pol(nir_cv2)
        daolpnir = transform_pol(daolpnir)
        gt = transform_gt(gt).squeeze(0) * 255
        gt_offset = transform_gt(gt_offset).squeeze(0) * 255

        # sample = {'image': color_image, 'aolp': aolp, 'dolp': dolp}
        # sample = {'image': color_image, 'range_view': daolp, "target": gt}
        sample = {'image': color_image_offset, 'range_view': nir_cv2, "target": gt_offset}

        # range_view = transform_image(range_view)
        # range_view = range_view[1]
        # range_view = range_view[None, :, :]
        # sample = torch.cat((color_image, range_view), 0)

        return sample


class VisNirDataset(Dataset):

    def __init__(self, root_dir, image_size, transform=None):
        """
        Arguments:.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.image_list = os.listdir(self.root_dir + "/vis/")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb_root = os.path.join(self.root_dir, "vis")
        nir_root = os.path.join(self.root_dir, "nir")
        target_root = os.path.join(self.root_dir, "labels_enc")

        img_path = os.path.join(rgb_root, self.image_list[idx])
        image = Image.open(img_path).convert('RGB')

        nir_path = os.path.join(nir_root, self.image_list[idx])
        nir = Image.open(nir_path).convert('L')

        target_path = os.path.join(target_root, self.image_list[idx])
        target = Image.open(target_path).convert('L')

        transform_nir = transforms.Compose(
            [
                transforms.Resize(((self.image_size[0], self.image_size[1])),
                                  transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ]
        )

        transform_image = transforms.Compose(
            [
                transforms.Resize(((self.image_size[0], self.image_size[1])),
                                  transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                #transforms.Normalize([0.4585, 0.4543, 0.4307], [0.2751, 0.2796, 0.2818])
            ]
        )

        transform_range = transforms.Compose(
            [
                transforms.Resize(((self.image_size[0], self.image_size[1])), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

        image = transform_image(image)
        range_view = transform_nir(nir)
        target = transform_range(target).squeeze(0) * 255

        sample = {"image": image, "range_view": range_view, "target": target}

        return sample

class MultiSpecCropDataset(Dataset):

    def __init__(self, root_dir, image_size, mean: Optional[List[float]], std: Optional[List[float]], split="rose2_haricot", band=5):
        """
        Arguments:.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = os.path.join(root_dir, split)
        self.image_size = image_size
        self.image_list = os.listdir(self.data_dir)
        self.band = band
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        folder_path = os.path.join(self.data_dir, self.image_list[idx])
        img_path = os.path.join(folder_path, "false.png")
        image = Image.open(img_path)
        width, height = image.size

        sem_annos = np.array(Image.open(os.path.join(folder_path, 'gt.png'))).astype(np.int32)
        sem = np.zeros((sem_annos.shape[0], sem_annos.shape[1]))
        sem[sem_annos[:, :, 1] != 0] = 2
        sem[sem_annos[:, :, 2] != 0] = 1

        im = io.imread(os.path.join(folder_path, "images.tiff"))

        range_view = Image.fromarray(im[:, :, self.band])
        sem = Image.fromarray(sem)

        transform_image = transforms.Compose(
            [
                transforms.Resize(((self.image_size[0], self.image_size[1])),
                                  transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

        transform_range = transforms.Compose(
            [
                transforms.Resize((self.image_size[0], self.image_size[1]), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

        if self.mean:
            transform_image = transforms.Compose(
                [
                    transform_image,
                    transforms.Normalize(mean=self.mean, std=self.std)
                ]
            )
        #range_view = skimage.transform.resize(im[:, :, self.band],(self.image_size[0], self.image_size[1]), order=1, mode="edge")
        #sem = skimage.transform.resize(sem,(self.image_size[0], self.image_size[1]), order=1, mode="edge")

        image = transform_image(image)
        range_view = transform_range(range_view)
        sem = transform_range(sem)

        sample = {"image": image.float(), "range_view": range_view.float(), "target": sem.float()}

        return sample
