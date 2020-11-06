import os
import re
import json
import pickle
import h5py
import random
import scipy
import numpy as np
import torch
import torch.nn.functional as F
from six.moves import cPickle
from pathlib import Path
from datetime import datetime
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
