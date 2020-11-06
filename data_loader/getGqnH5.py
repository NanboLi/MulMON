import sys
import torch
from torch.utils.data import Dataset
sys.path.append('../')
from utils import read_h5py
from data_loader.base_data_loader import BaseDataLoader


class GqnDataset(Dataset):
    """
    A general dataloder for the GQN data (.h5). The data support segmentation evaluation
    """
    def __init__(self, addr_root, data_file, num_slots=7, transform=None, background=True):
        """
        :param data_file: (abs_path) the root dir pf the dataset
        :param transform: (callable, optional): Optional transform to be applied on a sample.
        """
        self.addr_root = addr_root
        F = read_h5py(data_file)
        self.images = F['images']
        self.viewpoints = F['viewpoints']
        self.num_views = self.viewpoints.shape[1]
        self.K_slots = num_slots
        self.background = background
        self.transform = transform

    def __len__(self):
        return self.viewpoints.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx].astype('float32')
        if self.images[idx].dtype == 'uint8':
            image /= 255.0
        view_pts = self.viewpoints[idx].astype('float32')
        assert view_pts.ndim == 2, 'viewpoint data dimension bug, should be 2, but got {}'.format(view_pts.ndim)

        pre_sample = {'image': image,
                      'view_points': view_pts}

        if self.transform:
            assert isinstance(self.transform, list)
            for tfm in self.transform:
                pre_sample = tfm(pre_sample)
        gt = {
              'scn_id': torch.tensor([idx], requires_grad=False),
              'view_points': pre_sample['view_points'],
             }
        return pre_sample['image'], gt

    def extract_masks(self, scn_pack, masks, use_bg):
        raise NotImplementedError

    def normalise_viewpoints(self, item):
        raise NotImplementedError


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']  # [V, H, W, C]
        view_pts = sample['view_points']  # [V, D]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((0, 3, 1, 2))

        sample['image'] = torch.from_numpy(image)
        sample['view_points'] = torch.from_numpy(view_pts)
        return sample


class DataLoader(BaseDataLoader):
    def __init__(self, data_root, datafile_path, batch_size, shuffle=True, validation_split=0.0, num_workers=8,
                 num_slots=7, use_bg=True):
        trsfm = [
                 ToTensor(),
                ]
        self.dataset = GqnDataset(addr_root=data_root,
                                  data_file=datafile_path,
                                  num_slots=num_slots,
                                  transform=trsfm,
                                  background=use_bg)
        collate_fn = lambda x: tuple(zip(*x))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)


def distributed_loader(data_root, datafile_path, num_slots=7, use_bg=True):
    trsfm = [
        ToTensor(),
    ]
    return GqnDataset(addr_root=data_root,
                      data_file=datafile_path,
                      num_slots=num_slots,
                      transform=trsfm,
                      background=use_bg)
