import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
sys.path.append('../')
from utils import read_json
from data_loader.base_data_loader import BaseDataLoader


class ClevrMVDataset(Dataset):
    """
    Loading self-generated Clevr muti-view dataset.
    """
    def __init__(self, addr_root, data_file, num_slots=7, transform=None, background=True):
        """
        :param data_file: (abs_path) the root dir pf the dataset
        :param transform: (callable, optional): Optional transform to be applied on a sample.
        """
        self.addr_root = addr_root
        self.name = read_json(data_file)['info']
        self.data_base = read_json(data_file)['scenes']
        self.categories = read_json(data_file)['shape_gallery']
        self.num_views = read_json(data_file)['num_views']
        self.K_slots = num_slots
        self.background = background
        self.transform = transform

    def __len__(self):
        return len(self.data_base)

    def __getitem__(self, idx):
        item = self.data_base[idx]
        mydata = np.load(os.path.join(self.addr_root, item['data_path']))
        image = mydata['images'][..., :3].astype('float32')/255
        mask_pack_raw = mydata['masks']
        masks_obs, masks_clr, num_comps = self.extract_masks(item, mask_pack_raw, self.background)  # [V, H, W, K]
        view_pts = self.normalise_viewpoints(item)

        if masks_obs.ndim == 3:
            masks_obs = np.expand_dims(masks_obs, axis=-1)
        if masks_clr.ndim == 3:
            masks_clr = np.expand_dims(masks_clr, axis=-1)
        assert view_pts.ndim == 2, 'viewpoint data dimension bug, should be 2, but got {}'.format(view_pts.ndim)

        pre_sample = {'image': image,
                      'masks': masks_obs,
                      'masks_clr': masks_clr,
                      'num_comps': num_comps,
                      'view_points': view_pts}

        if self.transform:
            assert isinstance(self.transform, list)
            for tfm in self.transform:
                pre_sample = tfm(pre_sample)

        gt = {
              'scn_id': torch.tensor([idx], requires_grad=False),           # access by gt['scn_id'].item()
              'num_comps': pre_sample['num_comps'],
              'masks': pre_sample['masks'],
              'masks_clr': pre_sample['masks_clr'],
              'view_points': pre_sample['view_points'],
             }
        return pre_sample['image'], gt

    def extract_masks(self, scn_pack, masks, use_bg):
        """ Separating the raw mask data into two groups (masks w/o occlusions)
        :param scn_pack:  scn dict that contains information about the scene
        :param masks:  [V, H, W, N], where N <= K
        :param use_bg:  include background as object or not.
        """
        V, H, W, _ = masks.shape

        mv_masks = []
        mv_mclrs = []
        for v in range(self.num_views):
            m_occ_raw = masks[v, ..., 0]
            m_clr = masks[v, ..., 1:]

            if use_bg:
                depth_order = [0] + list(i + 1 for i in scn_pack['depth_orders'][v])
                m_clr = np.concatenate((np.ones_like(m_clr[..., :1]), m_clr), axis=-1)
                assert len(depth_order) == m_clr.shape[-1], 'depth order and m_clr size do not match'
                skip_bg = 0
            else:
                depth_order = scn_pack['depth_orders'][v]
                skip_bg = 1
            m_clr = np.stack([m_clr[..., i] for i in depth_order], axis=-1).astype('uint8')

            m_occ = []
            for vi in range(skip_bg, int(m_occ_raw.max()) + 1):
                m_occ.append(np.float32(m_occ_raw == vi))
            m_occ = np.stack(m_occ, axis=-1).astype('uint8')
            assert m_occ.shape[-1] == m_clr.shape[-1], \
                'depth order wrong {} '.format(scn_pack['mask_path'])

            # fill in slots with empty masks (nothing)
            N = m_occ.shape[-1]
            mask_occ_slots = np.zeros([H, W, self.K_slots], dtype='uint8')
            mask_clr_slots = np.zeros([H, W, self.K_slots], dtype='uint8')

            mask_occ_slots[..., :N] = m_occ
            mask_clr_slots[..., :N] = m_clr

            # record every single observations
            mv_masks.append(mask_occ_slots)
            mv_mclrs.append(mask_clr_slots)

        return np.stack(mv_masks, axis=0), np.stack(mv_mclrs, axis=0), N

    def normalise_viewpoints(self, item):
        """ view_pts standardization (not really necessary though) """
        cams = np.asarray(item['camera']['location'])
        assert cams.shape[0] == self.num_views, 'non-valid data: {}'.format(item['image_filename'])
        cams[:, 2] = 0.
        cams /= (np.linalg.norm(cams, axis=1, keepdims=True) + 1e-5)
        return (cams - np.asarray([[1., 0., 0.]])).astype('float32')


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']  # [V, H, W, C]
        masks = sample['masks']  # [V, H, W, K]
        masks_clr = sample['masks_clr']  # [V, H, W, K]
        num_objects = sample['num_comps']
        view_pts = sample['view_points']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((0, 3, 1, 2))

        # for masks:
        # numpy masks: V, H, W, K
        # torch masks: V, 1, H, W, K
        if masks.ndim == 4:
            masks = np.expand_dims(masks, axis=1)  # [V, 1, H, W, K]
        if masks_clr.ndim == 4:
            masks_clr = np.expand_dims(masks_clr, axis=1)  # [V, 1, H, W, K]
        n_comps_npy = np.int16(num_objects).reshape([-1, 1])

        sample['num_comps'] = torch.from_numpy(n_comps_npy)
        sample['image'] = torch.from_numpy(image)
        sample['masks'] = torch.from_numpy(masks)
        sample['masks_clr'] = torch.from_numpy(masks_clr)
        sample['view_points'] = torch.from_numpy(view_pts)
        return sample


class DataLoader(BaseDataLoader):
    def __init__(self, data_root, datafile_path, batch_size, shuffle=True, validation_split=0.0, num_workers=4,
                 num_slots=7, use_bg=True):
        trsfm = [
                 ToTensor(),
                ]
        self.dataset = ClevrMVDataset(addr_root=data_root,
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
    return ClevrMVDataset(addr_root=data_root,
                          data_file=datafile_path,
                          num_slots=num_slots,
                          transform=trsfm,
                          background=use_bg)

