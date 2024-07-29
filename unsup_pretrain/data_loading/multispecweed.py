import torch
import torchvision
import os
import numpy as np
from skimage import io
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.tenprint import print_tensor
import skimage


class MultiSpectralCropWeed(Dataset):

    def __init__(self, root_dir, image_size, split="rose2_haricot", band=5):
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

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        folder_path = os.path.join(self.data_dir, self.image_list[idx])
        img_path = os.path.join(folder_path, "false.png")
        image = Image.open(img_path)
        width, height = image.size

        # sem_annos = np.array(Image.open(os.path.join(folder_path, 'gt.png'))).astype(np.int32)
        # sem = np.zeros((sem_annos.shape[0], sem_annos.shape[1]))
        # sem[sem_annos[:, :, 1] != 0] = 2
        # sem[sem_annos[:, :, 2] != 0] = 1

        im = io.imread(os.path.join(folder_path, "images.tiff"))
        range_view = Image.fromarray(im[:, :, self.band])

        transform_image = transforms.Compose(
            [
                transforms.Resize(((self.image_size[0], self.image_size[1])), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.4422, 0.4400, 0.2145], std=[0.1684, 0.1434, 0.0600]),
            ]
        )

        transform_range = transforms.Compose(
            [
                transforms.Resize((self.image_size[0], self.image_size[1]), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

        #range_view = skimage.transform.resize(im[:, :, self.band],
        #                                   (self.image_size[0], self.image_size[1]), order=1, mode="edge")
        image = transform_image(image)
        range_view = transform_range(range_view)

        sample = {"image": image.float(), "range_view": range_view.float()}

        return sample


if __name__ == '__main__':
    dataset = MultiSpectralCropWeed(root_dir='/home/matt/data/roses',
                                           image_size=[114, 172])
    import matplotlib
    # print(matplotlib.get_backend())
    # matplotlib.use('Qt5Agg')
    # print(matplotlib.get_backend())
    fig = plt.figure()

    for sample in dataset:
        # print(matplotlib.get_backend())
        print(sample["image"].shape)
        print(sample["range_view"].shape)
        # print_tensor(sample["range_view"])
        # print_tensor(sample["image"])


