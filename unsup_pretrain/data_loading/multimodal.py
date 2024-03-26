import skimage.transform
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class MultimodalMaterialDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Arguments:.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        list_dir = os.path.join(root_dir, 'list_folder')
        with open(os.path.join(os.path.join(list_dir, "train" + '.txt')), "r") as f:
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

        color_name = os.path.join(color_dir, self.image_list[idx] + ".png")
        color_image = Image.open(color_name).convert('RGB')
        aolp_sin_name = os.path.join(aolp_sin_dir, self.image_list[idx] + ".npy")
        aolp_sin = np.load(aolp_sin_name)
        aolp_cos_name = os.path.join(aolp_cos_dir, self.image_list[idx] + ".npy")
        aolp_cos = np.load(aolp_cos_name)
        aolp = np.stack([aolp_sin, aolp_cos], axis=2)

        dolp_name = os.path.join(dolp_dir, self.image_list[idx] + ".npy")
        dolp = np.load(dolp_name)

        daolp = np.stack([aolp_sin, aolp_cos, dolp], axis=2)
        daolp = skimage.transform.resize(daolp, (256, 306), order=1, mode="edge")

        transform_image = transforms.Compose(
            [
                transforms.Resize((256, 306), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        transform_pol = transforms.Compose(
            [
                # transforms.Resize((256, 306), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

        color_image = transform_image(color_image)
        daolp = transform_pol(daolp)

        # sample = {'image': color_image, 'aolp': aolp, 'dolp': dolp}
        sample = {'image': color_image, 'daolp': daolp}

        # range_view = transform_image(range_view)
        # range_view = range_view[1]
        # range_view = range_view[None, :, :]
        # sample = torch.cat((color_image, range_view), 0)

        return sample


if __name__ == '__main__':
    multimodal_dataset = MultimodalMaterialDataset(root_dir='/home/matt/data/multimodal_dataset')

    fig = plt.figure()

    for i, sample in enumerate(multimodal_dataset):
        print(i, sample)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')

        if i == 3:
            plt.show()
            break
