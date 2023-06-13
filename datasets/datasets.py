import torch
import yaml
import torchvision
from torchvision.datasets import Cityscapes
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, cfg):
        super().__init__()
        # from cfg i can access to all my shit
        # as data path, data size and so on 
        self.cfg = cfg
        self.len = -1
        self.setup()
        self.loader = [self.train_dataloader(), self.val_dataloader()]

    def prepare_data(self):
        # Augmentations are applied using self.transform 
        # no data to download, for now everything is local 
        pass

    def setup(self, stage=None):

        self.mode = self.cfg['train']['mode']

        if stage == 'fit' or stage is None:
            self.data_train = CityData(self.cfg['data']['ft-path'], split='train',
                                       mode='fine',
                                       target_type='semantic')
            self.data_val = CityData(self.cfg['data']['ft-path'], split='val',
                                     mode='fine',
                                     target_type='semantic')
        return

    def train_dataloader(self):
        loader = DataLoader(self.data_train,
                            batch_size=self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            num_workers=self.cfg['train']['workers'],
                            pin_memory=True,
                            shuffle=True)
        self.len = self.data_train.__len__()
        return loader

    def val_dataloader(self):
        if self.mode == 'pt': pass
        loader = DataLoader(self.data_val,
                            batch_size=1,  # self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            num_workers=self.cfg['train']['workers'],
                            pin_memory=True,
                            shuffle=False)
        self.len = self.data_val.__len__()
        return loader

    def test_dataloader(self):
        pass


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
        transform = transforms.Compose(
            [
                transforms.Resize((128, 256), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ]
        )
        tra = {'image': transform(image), 'target': (transform(target).squeeze(0) * 255).type(torch.int64)}

        # return transformed['image'], transformed['mask']

        if len(tra['image'].shape) != 3:
            tra['image'] = tra['image'].unsqueeze(0)

        # tra['image'] = tra['image'].permute(1, 2, 0)
        # tra['mask'] = tra['mask'].permute(1, 2, 0).squeeze()
        return tra
    # torch.unsqueeze(transformed['mask'], 0)
