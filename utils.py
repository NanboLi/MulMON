"""
Utility functions. Some are adapted from other repositories.
@author: Nanbo Li
"""
import os
import json
import pickle
import h5py
import random
import numpy as np
import torch
from itertools import repeat
from collections import OrderedDict
import pdb


# ------ General utilities ------
def ensure_dir(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)


def read_json(fname):
    with open(fname, "r") as read_file:
        return json.load(read_file)


def write_json(content, fname):
    with open(fname, "w") as write_file:
        json.dump(content, write_file)


def read_h5py(fname):
    return h5py.File(fname, 'r')


def write_pick(content, fname):
    pickle_out = open(fname, "wb")
    pickle.dump(content, pickle_out)
    pickle_out.close()


def load_pickle(fname):
    pickle_in = open(fname, "rb")
    return pickle.load(pickle_in)


# ------ Training utitiles ------
def inf_loop(data_loader):
    """wrapper function for endless data loader"""
    for loader in repeat(data_loader):
        yield from loader


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_trained_mp(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location='cpu')['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


# ------ numpy/pytorch utitiles ------
def numpify(tensor):
    return tensor.detach().cpu().numpy()


def iou_pair(outputs: torch.Tensor, labels: torch.Tensor):
    assert outputs.dim() == 2, 'expected [H, W], got {}'.format(outputs.size())
    assert outputs.dim() == 2, 'expected [H, W], got {}'.format(labels.size())
    assert outputs.dtype == torch.float32
    assert labels.dtype == torch.float32

    intersection = (outputs > 0.9) & (labels > 0.9)
    union = (outputs > 0.9) | (labels > 0.9)

    # union = (outputs + labels > 0.8).float().sum(())  # Will be zzero if both are 0
    iou = (intersection.float().sum() + 1e-6) / (union.float().sum() + 1e-6)
    iou = iou if iou > 0.05 else iou * 0.0
    return iou


def iou_scn(x, gt, threshold=1.0):
    """
    outputs: [K, H, W]
    labels:  [K, H, W]
    """
    assert x.dim() == 3
    assert gt.dim() == 3

    len_x = x.size(0)
    len_gt = gt.size(0)
    assert len_x == len_gt

    M = []
    for k in range(len_x):
        M.append(iou_pair(x[k], gt[k]))
    iou_scores = torch.stack(M, dim=0)

    # iou_scores = torch.stack(M, dim=0)
    if threshold < 1.0:
        return (iou_scores >= threshold).type(x.dtype).mean()
    else:
        return iou_scores.mean()


def iou_scn_unmatch(x, gt, threshold=1.0):
    """
    outputs: [K, H, W]
    labels:  [K, H, W]
    """
    assert x.dim() == 3
    assert gt.dim() == 3

    len_x = x.size(0)
    len_gt = gt.size(0)

    M = []
    for g in range(len_gt):
        assign = []
        for k in range(len_x):
            # USE ONE OF THE METRIC AS OBJECTIVE TO MATCH pred & gt
            assign.append(iou_pair(x[k], gt[g]))
        M.append(torch.tensor(assign))
    M = torch.stack(M, dim=0)
    # pdb.set_trace()
    iou_assign, m_indices = torch.max(M, dim=1)
    if threshold < 1.0:
        return (iou_assign >= threshold).type(x.dtype).mean(), m_indices.tolist()
    else:
        return iou_assign.mean(), m_indices.tolist()


def match_or_compute_segmentation_iou(m_preds, m_gts, num_comps, match_list=None, threshold=1.0):
    """ If the 'match_list' is provided, then we return the mIoU scores. Otherwise, this function does Hungarian-style
    matching and then return a match list.
    :param m_preds: [B, V, K, H, W] tensor
    :param m_gts: [B, V, K1, H, W] tensor
    :param num_comps: [B, V] the number of objects in the scene (given as GT)
    """
    assert m_preds.dim() == m_gts.dim()
    K = m_preds.size(2)
    B, V, K1, H, W = m_gts.size()
    assert K >= K1, 'K is smaller than the number of objects, segmentation evaluations should be disabled'

    assert num_comps.shape[0] == B
    num_comps = num_comps.reshape((B * V,)).tolist()

    m_preds = torch.zeros_like(m_preds).scatter(2, torch.max(m_preds, dim=2, keepdim=True)[1], 1)
    m_preds = m_preds.reshape(B * V, K, H, W)
    m_gts = m_gts.reshape(B * V, K1, H, W).type(m_preds.dtype)
    N = m_preds.size(0)
    assert m_gts.size(0) == N

    iou_list = []
    if match_list is None:
        match_list = []
        for i in range(N):
            miou_scn_val, gt_to_out = iou_scn_unmatch(m_preds[i], m_gts[i, :num_comps[i]], threshold=threshold)
            iou_list.append(numpify(miou_scn_val))
            match_list.append(gt_to_out)
        return iou_list, match_list
    else:
        for i in range(N):
            miou_scn_val = iou_scn(m_preds[i, match_list[i]], m_gts[i, :num_comps[i]], threshold=threshold)
            iou_list.append(numpify(miou_scn_val))
        return iou_list, None


###############################
# Disentanglement related
###############################

shape_code = {
    "_background_": [0, 0, 0, 0, 0, 0, 1],
    "cube":         [0, 0, 0, 0, 0, 1, 0],
    "owl":          [0, 0, 0, 0, 1, 0, 0],
    "duck":         [0, 0, 0, 1, 0, 0, 0],
    "mug":          [0, 0, 1, 0, 0, 0, 0],
    "horse":        [0, 1, 0, 0, 0, 0, 0],
    "teapot":       [1, 0, 0, 0, 0, 0, 0]
}
color_code = {
    "gray":   [87, 87, 87],
    "red":    [173, 35, 35],
    "blue":   [42, 75, 215],
    "green":  [29, 105, 20],
    "brown":  [129, 74, 25],
    "purple": [129, 38, 192],
    "cyan":   [41, 208, 208],
    "yellow": [255, 238, 51]
}
mat_code = {
    'rubber': [0, 1],
    'metal':  [1, 0],
}


def save_latents_for_eval(z_v_out, z_out, scn_indices, qry_views, gt_scenes_meta,
                          out_dir=None,
                          save_count=0):
    B = len(scn_indices)
    GTs = []
    for s_count, sid in enumerate(scn_indices):
        # correct objects' permuations
        obj_to_gt_idx = []
        # for v in range(len(z_v_out)):
        #     obj_to_gt_idx.append(gt_scenes_meta[sid]['depth_orders'][v])
        for v in qry_views:
            obj_to_gt_idx.append(gt_scenes_meta[sid]['depth_orders'][v])
        z_v_out = np.stack(z_v_out, axis=0)

        actual_idx = save_count + s_count
        base_dir_name = os.path.basename(out_dir)
        save_to = None
        if out_dir:
            ensure_dir(out_dir)
            save_to = os.path.join(out_dir, 'zout_{:06d}'.format(actual_idx))
            np.savez(save_to,
                     out_z_v=z_v_out,
                     out_z=z_out[s_count])

        s = {
            'z_path': '{}/{}'.format(base_dir_name, os.path.basename(save_to)),
            'obj_to_gt_map': obj_to_gt_idx,
            'objects': gt_scenes_meta[sid]['objects'],
            'query_views': qry_views
        }
        GTs.append(s)
    return GTs