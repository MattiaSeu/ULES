import skimage.transform
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io


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

        color_dir = os.path.join(self.root_dir, 'polL_color')
        aolp_sin_dir = os.path.join(self.root_dir, 'polL_aolp_sin')
        aolp_cos_dir = os.path.join(self.root_dir, 'polL_aolp_cos')
        dolp_dir = os.path.join(self.root_dir, 'polL_dolp')
        nir_dir = os.path.join(self.root_dir, 'NIR_warped')

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
        daolp = skimage.transform.resize(daolp, (self.image_size[0], self.image_size[1]), order=1, mode="edge")
        dolp = skimage.transform.resize(dolp, (self.image_size[0], self.image_size[1]), order=1, mode="edge")

        nir_left_offset = 192
        nir_name = os.path.join(nir_dir, self.image_list[idx] + ".png")
        nir = Image.open(nir_name).convert('LA')
        import cv2
        nir_cv2 = cv2.imread(nir_name, -1)
        # plt.imshow(nir_cv2)
        # plt.show()
        # nir_cv2 = io.imread(nir_name)
        nir_cv2 = skimage.transform.resize(nir_cv2[:, nir_left_offset:],
                                           (self.image_size[0], self.image_size[1]), order=1, mode="edge")
        # nir_cv2 = Image.fromarray(nir_cv2[:, nir_left_offset:])
        # nir_cv2 = nir_cv2.convert('L')
        w, h = color_image.size
        color_image_offset = color_image.crop((nir_left_offset, 0, w, h))

        transform_image = transforms.Compose(
            [
                transforms.Resize((self.image_size[0], self.image_size[1]), transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.30553093, 0.29185508, 0.3206072), std=(0.31198422, 0.31180399, 0.32578236)),
            ]
        )

        transform_pol = transforms.Compose(
            [
                # transforms.Resize((self.image_size[0], self.image_size[1]), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

        transform_nir = transforms.Compose(
            [
                transforms.Resize((self.image_size[0], self.image_size[1]), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

        color_image_full = transform_image(color_image)
        color_image_offset = transform_image(color_image_offset)
        daolp = transform_pol(daolp)
        dolp = transform_pol(dolp)
        nir = transform_nir(nir)
        nir_cv2 = transform_pol(nir_cv2)

        # when using nir capture is limited on the left so we remove it from the other modalities

        # sample = {'image': color_image_full, 'aolp': aolp, 'dolp': dolp}
        # sample = {'image': color_image_full, 'daolp': daolp}
        sample = {'image': color_image_offset.float(),
                  'range_view': nir_cv2.float()}
        # sample = {'image': color_image_full, 'range_view': dolp}

        # range_view = transform_image(range_view)
        # range_view = range_view[1]
        # range_view = range_view[None, :, :]
        # sample = torch.cat((color_image, range_view), 0)

        return sample


if __name__ == '__main__':
    dataset = MultimodalMaterialDataset(root_dir='/home/matt/data/multimodal_dataset', split="train",
                                                   image_size=[130, 130])
                                                   # image_size=[256, 306])
    from utils.tenprint import print_tensor

    import matplotlib

    print(matplotlib.get_backend())
    # matplotlib.use('Qt5Agg')
    # print(matplotlib.get_backend())
    fig = plt.figure()

    for sample in dataset:
        print(matplotlib.get_backend())
        # print(sample["image"].shape)
        # print(sample["range_view"].shape)
        print_tensor(sample["range_view"])
        print_tensor(sample["image"])
