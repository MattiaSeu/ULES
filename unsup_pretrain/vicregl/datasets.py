# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets

from transforms import MultiCropTrainDataTransform, MultiCropValDataTransform
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from skimage import io
import cv2
import skimage
import argparse

IMAGENET_NUMPY_PATH = "/private/home/abardes/datasets/imagenet1k/"
IMAGENET_PATH = "/home/matt/data/imgfiles/"
CITYSCAPE_PATH = "~/data/cityscapes/"
KITTI_PATH = "/home/matt/data/kitti_sem/"
MULTISPEC_PATH = "/home/matt/data/roses/"
VISNIR_PATH = "/home/matt/data/vis-nir/"
KYOTO_PATH = "/home/matt/data/multimodal_dataset/"

class MultiSpectralCropWeed(Dataset):

    def __init__(self, root_dir, image_size, split="rose2_haricot", band=5, transform=None):
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
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        folder_path = os.path.join(self.data_dir, self.image_list[idx])
        img_path = os.path.join(folder_path, "false.png")
        range_view = None
        image = Image.open(img_path)
        width, height = image.size

        im = io.imread(os.path.join(folder_path, "images.tiff"))
        normalized_im = (im[:, :, self.band] / 65535.0 * 255).astype(np.uint8)

        range_view = Image.fromarray(normalized_im).convert("L")

        transformed = self.transform(image, range_view)
        return transformed

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


        transformed = self.transform(image, nir)
        # return tuple([transformed[0][0], transformed[1]])
        return transformed

class MultimodalMaterialDataset(Dataset):

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

        self.transform(color_image_offset, nir)

        transformed = self.transform(color_image_offset, nir)
        return transformed


class KittiRangeDataset(Dataset):

    def __init__(self, root_dir, image_size, split: str, transform=None):
        """
        Arguments:.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.image_list = os.listdir(self.root_dir + "/rgb/")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb_root = os.path.join(self.root_dir, "rgb")
        # target_root = os.path.join(self.root_dir, "gray")
        range_root = os.path.join(self.root_dir, "sequences")

        img_path = os.path.join(rgb_root, self.image_list[idx])

        image = Image.open(img_path).convert('RGB')

        seq_name_list = self.image_list[idx].split(sep=".")[0].split(sep="_")

        # target_path = os.path.join(target_root, f"{seq_name_list[0]}_{seq_name_list[1]}.png")

        # target = Image.open(target_path).convert('L')

        range_view = Image.open(range_root + "/" + seq_name_list[0] + "/range_projection/" +
                                seq_name_list[1] + ".png").convert("L")

        transform_image = transforms.Compose(
            [
                transforms.Resize(((self.image_size[0], self.image_size[1])), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

        transform_range = transforms.Compose(
            [
                transforms.Resize(((self.image_size[0], self.image_size[1])), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

        # image = transform_image(image)
        # target = transform_range(target).squeeze(0) * 255
        # range_view = transform_image(range_view)

        transfomed = self.transform(image, range_view)

        image = self.transform(image)
        range_view = self.transform(range_view)
        sample = {"image": image, "range_view": range_view}  # , "target": target.type(torch.int64)}

        # return sample
        return image , range_view


class ImageNetNumpyDataset(Dataset):
    def __init__(self, img_file, labels_file, size_dataset=-1, transform=None):
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform

    def get_img(self, path, transform):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        if transform is not None:
            img = transform(img)
        return img

    def __getitem__(self, i):
        img = self.get_img(self.samples[i], self.transform)
        lab = self.labels[i]
        return img, lab

    def __len__(self):
        return len(self.samples)


def build_loader(args, is_train=True):
    dataset = build_dataset(args, is_train)

    batch_size = args.batch_size
    if (not is_train) and args.val_batch_size == -1:
        batch_size = args.batch_size

    sampler = torch.utils.data.RandomSampler(dataset)
    # per_device_batch_size = batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        # sampler=sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )

    return loader#, sampler


def build_dataset(args, is_train=True):
    transform = build_transform(args, is_train=is_train)

    if args.dataset == "imagenet1k":
        args.num_classes = 10

        if args.dataset_from_numpy:
            root = IMAGENET_NUMPY_PATH
            prefix = "train" if is_train else "val"
            images_path = os.path.join(root, f"{prefix}_images.npy")
            labels_path = os.path.join(root, f"{prefix}_labels.npy")
            dataset = ImageNetNumpyDataset(
                images_path, labels_path, transform=transform
            )
        else:
            root = IMAGENET_PATH
            prefix = "train" if is_train else "val"
            path = os.path.join(root, "train")
            dataset = datasets.ImageFolder(path, transform)
    if args.dataset == "kittisem":
        args.num_classes = 12
        root = KITTI_PATH
        split = "train" if is_train else "val"
        dataset = KittiRangeDataset(root, split=split, image_size=[224, 224], transform=transform)
    if args.dataset == "kyotomaterial":
        args.num_classes = 20
        root = KYOTO_PATH
        split = "train" if is_train else "val"
        dataset = KittiRangeDataset(root, split=split, image_size=[224, 224], transform=transform)
    if args.dataset == "roses":
        args.num_classes = 3
        root = MULTISPEC_PATH
        split = "roses23" if is_train else None
        dataset = MultiSpectralCropWeed(root, split=split, image_size=[224, 224], transform=transform)
        TEST_SIZE = 0.2
        SEED = 42

        # generate indices: instead of the actual data we pass in integers instead
        train_indices, test_indices = train_test_split(
            range(len(dataset)),
            test_size=TEST_SIZE,
            random_state=SEED
        )

        # generate subset based on indices
        dataset = Subset(dataset, train_indices)
    if args.dataset == "visnir":
            args.num_classes = 23
            root = VISNIR_PATH
            dataset = VisNirDataset(root, image_size=[224, 224], transform=transform)
            TEST_SIZE = 0.2
            SEED = 42

            # generate indices: instead of the actual data we pass in integers instead
            train_indices, test_indices = train_test_split(
                range(len(dataset)),
                test_size=TEST_SIZE,
                random_state=SEED
            )

            # generate subset based on indices
            dataset = Subset(dataset, train_indices)
    return dataset


def build_transform(args, is_train=True):
    transform_args = {
        "size_crops": args.size_crops,
        "num_crops": args.num_crops,
        "min_scale_crops": args.min_scale_crops,
        "max_scale_crops": args.max_scale_crops,
        "return_location_masks": True,
        "no_flip_grid": args.no_flip_grid,
    }
    if is_train:
        transform = MultiCropTrainDataTransform(**transform_args)
    else:
        transform = MultiCropValDataTransform(**transform_args)

    return transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretraining with VICRegL", add_help=False)
    parser.add_argument("--size-crops", type=int, nargs="+", default=[224, 96])
    parser.add_argument("--num-crops", type=int, nargs="+", default=[2, 6])
    parser.add_argument("--min_scale_crops", type=float, nargs="+", default=[0.4, 0.08])
    parser.add_argument("--max_scale_crops", type=float, nargs="+", default=[1, 0.4])
    parser.add_argument("--no-flip-grid", type=int, default=1)
    args = parser.parse_args()
    transform = build_transform(args, is_train=True)
    root = VISNIR_PATH
    # split = "roses23"
    dataset = VisNirDataset(root, image_size=[224, 224], transform=transform)
    split = "train"
    # dataset = MultimodalMaterialDataset(root, split = split, image_size=[224, 224], transform=transform)

    TEST_SIZE = 0.2
    SEED = 42

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=TEST_SIZE,
        random_state=SEED
    )

    # generate subset based on indices
    dataset = Subset(dataset, train_indices)

    from utils import print_tensor

    for sample in dataset:

        # for j in range(len(sample[0][0])):
            # print_tensor(sample[0][0][j])
            # print_tensor(sample[0][1][j])
        for j in range(len(sample[0])):
            print_tensor(sample[0][j])
