import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class KittiRangeDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Arguments:.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.image_list[idx])

        image = Image.open(img_name).convert('RGB')

        seq_name_list = self.image_list[idx].split(sep=".")[0].split(sep="_")

        range_view = Image.open("/home/matt/data/kitti_sem/train/sequences/" + seq_name_list[0] + "/range_projection/" +
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
        range_view = transform_image(range_view)

        range_view = range_view[1]
        range_view = range_view[None, :, :]
        sample = torch.cat((image, range_view), 0)

        return sample


if __name__ == '__main__':
    kitti_dataset = KittiRangeDataset(root_dir='/home/matt/data/kitti_sem/train/rgb/')

    fig = plt.figure()

    for i, sample in enumerate(kitti_dataset):
        print(i, sample)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')

        if i == 3:
            plt.show()
            break
