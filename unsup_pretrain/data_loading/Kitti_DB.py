import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


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
                                seq_name_list[1] + ".png").convert("LA")

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

        image = transform_image(image)
        # target = transform_range(target).squeeze(0) * 255
        range_view = transform_image(range_view)

        range_view = range_view[1]
        range_view = range_view[None, :, :]
        sample = {"image": image, "range_view": range_view}  # , "target": target.type(torch.int64)}

        return sample
