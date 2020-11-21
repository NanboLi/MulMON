# -*- coding: utf-8 -*-
"""
Visualisation utilities. 
@author: Nanbo Li
"""
import os
import imageio
from PIL import Image, ImageEnhance
import subprocess
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from skimage.transform import resize
import torch
from torchvision.utils import make_grid


def enhance_save_single_image(img, save_fname, out_size=None):
    if out_size:
        img = resize(img, out_size, anti_aliasing=True)
    if img.dtype != 'uint8':
        img = (img * 255).astype(np.uint8)
    pimg = Image.fromarray(img)
    enh={
        'bright': ImageEnhance.Brightness(pimg),
        'contra': ImageEnhance.Contrast(pimg),
    }
    pimg = enh['contra'].enhance(1.25)
    pimg = enh['bright'].enhance(1.25)
    pimg.save(save_fname)


def torch_save_image_enhanced(tensor, filename, nrow=8, padding=2,
                              normalize=False,
                              range=None,
                              scale_each=False,
                              pad_value=0,
                              enhance=False):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    if enhance:
        enh = {
            'bright': ImageEnhance.Brightness(im),
            'contra': ImageEnhance.Contrast(im),
        }
        im = enh['contra'].enhance(1.2)
        im = enh['bright'].enhance(1.2)
    im.save(filename)


def save_single_image(img, save_fname, out_size=None):
    if out_size:
        img = resize(img, out_size, anti_aliasing=True)
    imageio.imwrite(save_fname, img)


def save_uncertainty_plots(umap, save_to, cmap='jet'):
    plt.imshow(umap)
    plt.set_cmap(cmap)
    plt.tight_layout()
    plt.clim(0., 1.7)
    plt.colorbar()
    plt.savefig(save_to)
    plt.close()


def save_dorder_plots(img, K_comps=7, cmap='hsv'):
    fun = {
        'hsv': lambda t: cm.hsv(colors.Normalize(vmin=0, vmax=K_comps)(t), bytes=True),
        'Set1': lambda t: cm.Set1(colors.Normalize(vmin=0, vmax=K_comps)(t), bytes=True),
        'Set2': lambda t: cm.Set2(colors.Normalize(vmin=0, vmax=K_comps)(t), bytes=True)
    }
    return fun[cmap](K_comps-img)
    # plt.imshow(fun(K_comps-img))
    # plt.tight_layout()
    # plt.savefig(save_to)
    # plt.close()


def map_val_colors(img, v_min=0.0, v_max=1.0, cmap='hot'):
    fun = {
           'hot': lambda t: cm.afmhot(colors.Normalize(vmin=v_min, vmax=v_max)(t), bytes=True),
           'jet': lambda t: cm.jet(colors.Normalize(vmin=v_min, vmax=v_max)(t), bytes=True),
           'Greys': lambda t: cm.Greys(colors.Normalize(vmin=v_min, vmax=v_max)(t), bytes=True),
           'Blues': lambda t: cm.Blues(colors.Normalize(vmin=v_min, vmax=v_max)(t), bytes=True),
    }
    return fun[cmap](img)


def test_vis(cfg):
    DIR = os.path.join(cfg.vis_train_dir, 'epoch_'+str(cfg.num_epochs))
    npys = sorted(glob.glob(os.path.join(DIR, '*.npy')))

    npy = np.random.choice(npys)
    arr = np.load(npy)
    unpacked = []
    for i in range(arr.shape[0]):
        if i==0:
            unpacked.append(arr[i, ...])
        else:
            unpacked.append(np.int8(arr[i, ...]*255))

    multi_plots(unpacked, 1, arr.shape[0],
                figsize=[2+len(unpacked)*4, 4])
    return None


def save_images_grid(imgs_, nrows, ncols, fig_size=4, save_to=None):
    assert type(nrows) == int
    assert type(nrows) == int
    num_images = len(imgs_)
    assert nrows*ncols == num_images
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * fig_size, ncols * fig_size))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
          ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
          ax.set_xticks([])
          ax.set_yticks([])
    else:
        ax.axis('off')

    if save_to:
        plt.tight_layout()
        plt.savefig(save_to)
        plt.close()


def show_images_grid(imgs_, num_images=25, fig_size=4, save_to=None):
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * fig_size, ncols * fig_size))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
          ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
          ax.set_xticks([])
          ax.set_yticks([])
    else:
        ax.axis('off')

    if save_to:
        plt.tight_layout()
        plt.savefig(save_to)
        plt.close()


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.
    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str + ' ' + output_gif
    subprocess.call(str1, shell=True)
