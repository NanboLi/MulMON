# -*- coding: utf-8 -*-
"""
Greating demos --- scene factorisation, novel-view synthesis, and disentanglement.
@author: Nanbo Li
"""
import sys
import os
import random
from attrdict import AttrDict
import numpy as np
import argparse
import torch

# set project search path
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

import utils
from config import CONFIG
import pdb


# ------------------------- respecify important flags ------------------------
def running_cfg(cfg):
    ###########################################
    # Config i/o path
    ###########################################
    if cfg.DATA_TYPE == 'gqn_jaco':
        image_size = [64, 64]
        CLASSES = ['_background_', 'jaco', 'generic']
        cfg.v_in_dim = 7
        cfg.max_sample_views = 6
        data_dir = cfg.DATA_ROOT
        assert os.path.exists(data_dir)
        train_data_filename = os.path.join(data_dir, 'gqn_jaco', 'gqn_jaco_train.h5')
        test_data_filename = os.path.join(data_dir, 'gqn_jaco', 'gqn_jaco_test.h5')
        assert os.path.isfile(train_data_filename)
        assert os.path.isfile(test_data_filename)
    elif cfg.DATA_TYPE == 'clevr_mv':
        image_size = [64, 64]
        CLASSES = ['_background_', 'cube', 'sphere', 'cylinder']
        cfg.v_in_dim = 3
        cfg.max_sample_views = 6
        data_dir = cfg.DATA_ROOT
        assert os.path.exists(data_dir)
        train_data_filename = os.path.join(data_dir, 'clevr_mv', 'clevr_mv_train.json')
        test_data_filename = os.path.join(data_dir, 'clevr_mv', 'clevr_mv_test.json')
        assert os.path.isfile(train_data_filename)
        assert os.path.isfile(test_data_filename)
    elif cfg.DATA_TYPE == 'clevr_aug':
        image_size = [64, 64]
        CLASSES = ['_background_', 'diamond', 'duck', 'mug', 'horse', 'dolphin']
        cfg.v_in_dim = 3
        cfg.max_sample_views = 6
        data_dir = cfg.DATA_ROOT
        assert os.path.exists(data_dir)
        train_data_filename = os.path.join(data_dir, 'clevr_aug', 'clevr_aug_train.json')
        test_data_filename = os.path.join(data_dir, 'clevr_aug', 'clevr_aug_test.json')
        assert os.path.isfile(train_data_filename)
        assert os.path.isfile(test_data_filename)
    # ------------------- For your customised CLEVR -----------------------
    elif cfg.DATA_TYPE == 'your-clevr':
        image_size = [64, 64]
        CLASSES = ['_background_', 'xxx']
        cfg.v_in_dim = 3
        cfg.max_sample_views = 6
        data_dir = cfg.DATA_ROOT
        assert os.path.exists(data_dir)
        train_data_filename = os.path.join(data_dir, 'your-clevr', 'your-clevr_train.json')
        test_data_filename = os.path.join(data_dir, 'your-clevr', 'your-clevr_test.json')
        assert os.path.isfile(train_data_filename)
        assert os.path.isfile(test_data_filename)
    # ------------------- For your customised CLEVR -----------------------
    else:
        raise NotImplementedError

    cfg.view_dim = cfg.v_in_dim

    # log directory
    ckpt_base = cfg.ckpt_base
    if not os.path.exists(ckpt_base):
        os.mkdir(ckpt_base)

    # model savedir
    check_dir = os.path.join(ckpt_base, '{}_log/'.format(cfg.arch))
    assert os.path.exists(check_dir)
    # os.mkdir(check_dir)

    # output prediction dir
    out_dir = os.path.join(check_dir, cfg.output_dir_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # saved model dir
    save_dir = os.path.join(check_dir, 'saved_models/')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # generated sample dir  (for testing generation)
    generated_dir = os.path.join(check_dir, 'generated_{:02d}/'.format(cfg.num_vq_show))
    if not os.path.exists(generated_dir):
        os.mkdir(generated_dir)

    if cfg.resume_path is not None:
        assert os.path.isfile(cfg.resume_path)
    elif cfg.resume_epoch is not None:
        resume_path = os.path.join(save_dir,
                                   'checkpoint-epoch{}.pth'.format(cfg.resume_epoch))
        assert os.path.isfile(resume_path)
        cfg.resume_path = resume_path

    cfg.DATA_DIR = data_dir
    cfg.test_data_filename = test_data_filename
    cfg.check_dir = check_dir
    cfg.save_dir = save_dir
    cfg.generated_dir = generated_dir
    cfg.output_dir = out_dir

    cfg.image_size = image_size
    cfg.CLASSES = CLASSES
    cfg.num_classes = len(CLASSES)

    return cfg


# ---------------------------- main function -----------------------------
def run_demo(CFG):
    if 'GQN' in CFG.arch:
        from models.baseline_gqn import GQN as ScnModel
        print(" --- Arch: GQN ---")
    elif 'IODINE' in CFG.arch:
        from models.baseline_iodine import IODINE as ScnModel
        print(" --- Arch: IODINE ---")
    elif 'MulMON' in CFG.arch:
        from models.mulmon import MulMON as ScnModel
        print(" --- Arch: MulMON ---")
    else:
        raise NotImplementedError

    # --- model to be evaluated ---
    scn_model = ScnModel(CFG)
    torch.cuda.set_device(CFG.gpu)

    if CFG.seed is None:
        CFG.seed = random.randint(0, 1000000)
    utils.set_random_seed(CFG.seed)

    if CFG.resume_epoch is not None:
        state_dict = utils.load_trained_mp(CFG.resume_path)
        scn_model.load_state_dict(state_dict, strict=True)
    scn_model.cuda(CFG.gpu)
    scn_model.eval()

    if 'gqn' in CFG.DATA_TYPE:
        # if 'h5' data is used (by default), otherwise, use json loader
        from data_loader.getGqnH5 import DataLoader
        # from data_loader.getGqnData import DataLoader
    elif 'clevr' in CFG.DATA_TYPE:
        from data_loader.getClevrMV import DataLoader
    else:
        raise NotImplementedError

    eval_dataloader = DataLoader(CFG.DATA_ROOT,
                                 CFG.test_data_filename,
                                 batch_size=CFG.batch_size,
                                 shuffle=True,
                                 use_bg=CFG.use_bg)

    vis_eval_dir = CFG.generated_dir
    if not os.path.exists(vis_eval_dir):
        os.mkdir(vis_eval_dir)

    # --- running on ---
    count_total_samples = 0
    num_batches = min(CFG.test_batch, len(eval_dataloader))
    for batch_id, (images, targets) in enumerate(eval_dataloader):
        if batch_id >= num_batches:
            break
        # images, targets = next(iter(eval_dataloader))
        images = list(image.cuda(CFG.gpu).detach() for image in images)
        targets = [{k: v.cuda(CFG.gpu).detach() for k, v in t.items()} for t in targets]

        if batch_id >= CFG.vis_batch:
            vis_eval_dir = None
        print("  predicting on batch: {}/{}".format(batch_id+1, num_batches))
        test_out = scn_model.predict(images, targets,
                                     save_sample_to=vis_eval_dir,
                                     save_start_id=count_total_samples,
                                     vis_train=False)

        B = len(images)
        V = targets[0]['view_points'].shape[0]

        # ----- traverse viewpoints to see 3D -----
        if CFG.traverse_v:
            # <<<<<<<<<< Load your own viewpoint trajectories (here we show) >>>>>>>>>>
            v_pts = torch.from_numpy(np.load(os.path.join(CFG.check_dir, 'interp_views.npy'))).cuda(CFG.gpu)
            select_views = torch.arange(72) * 5
            v_pts = v_pts[select_views, :]
            v_pts = v_pts.expand((B,) + v_pts.size()).float()

            scn_model.v_travel(test_out['lmbda'],
                               v_pts=v_pts,
                               save_sample_to=CFG.output_dir,
                               save_start_id=batch_id * CFG.batch_size)

        # ----- traverse 3D latents -----
        if CFG.traverse_z:
            v_pts = torch.stack([tar['view_points'] for tar in targets], dim=0).type(images[0].dtype)
            v_pts = torch.stack([v_pts[:, vid, :] for vid in test_out['query_views']], dim=1).cuda(CFG.gpu)
            z_3d = torch.from_numpy(test_out['3d_latents']).cuda(CFG.gpu)[:, 0, ...]

            # z_3d = torch.from_numpy(test_out['3d_latents']).cuda(CFG.gpu)
            scn_model.z_travel(z_3d,
                               v_pts=v_pts,
                               limit=4.0, int_step_size=0.2,
                               save_sample_to=CFG.output_dir,
                               save_start_id=batch_id * CFG.batch_size)

        count_total_samples += len(images)


def main(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ScnModel',
                        help="architecture name or model nickname")
    parser.add_argument('--datatype', type=str, default='clevr_mv',
                        help="one of [clevr_mv, clevr_aug, gqn-jaco]")
    parser.add_argument('--batch_size', default=16, type=int, metavar='N',
                        help='number of data samples of a minibatch')
    parser.add_argument('--test_batch', default=10000, type=int, metavar='N',
                        help='run model on only the first [N] batch of the data set')
    parser.add_argument('--vis_batch', default=1, type=int, metavar='N',
                        help='visualise only the first [N] batch and save to the generated dir')
    parser.add_argument('--analyse_batch', default=1, type=int, metavar='N',
                        help='save and analyse only the first [N] batch latents and ig_estimation')
    parser.add_argument('--work_mode', type=str, default='testing', help="model's working mode")
    parser.add_argument('--resume_epoch', default=500, type=int, metavar='N',
                        help='resume weights from [N]th epochs')
    parser.add_argument('--output_name', default=None, type=str,
                        help='save the prediction output to the specified dir')
    parser.add_argument('--gpu', default=0, type=int, help='specify id of gpu to use')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    # Model spec
    parser.add_argument('--num_slots', default=7, type=int, help='(maximum) number of component slots')
    parser.add_argument('--temperature', default=0.0, type=float,
                        help='spatial scheduler increase rate, the hotter the faster coeff grows')
    parser.add_argument('--latent_dim', default=16, type=int, help='size of the latent dimensions')
    parser.add_argument('--view_dim', default=5, type=int, help='size of the viewpoint latent dimensions')
    parser.add_argument('--min_sample_views', default=1, type=int, help='mininum allowed #views for scene learning')
    parser.add_argument('--max_sample_views', default=5, type=int, help='maximum allowed #views for scene learning')
    parser.add_argument('--num_vq_show', default=5, type=int, help='#views selected for visualisation')
    parser.add_argument('--pixel_sigma', default=0.1, type=float, help='loss strength item')
    parser.add_argument('--num_mc_samples', default=10, type=int, help='monte carlo samples for uncertainty estimation')
    parser.add_argument('--kl_latent', default=1.0, type=float, help='loss strength item')
    parser.add_argument('--kl_spatial', default=1.0, type=float, help='loss strength item')
    parser.add_argument('--exp_attention', default=1.0, type=float, help='loss strength item')
    parser.add_argument('--query_nll', default=1.0, type=float, help='loss strength item')
    parser.add_argument('--exp_nll', default=1.0, type=float, help='loss strength item')

    parser.add_argument("--use_bg", default=False, help="treat background also an object",
                        action="store_true")
    parser.add_argument("--traverse_v", default=False, help="traverse latent dimensions to see v disentanglement",
                        action="store_true")
    parser.add_argument("--traverse_z", default=False, help="traverse latent dimensions to see z disentanglement",
                        action="store_true")

    parser.add_argument("-i", '--input_dir', required=True, help="path to the input data for the model to read")
    parser.add_argument("-o", '--output_dir', required=True, help="destination dir for the model to write out results")
    args = parser.parse_args()

    ###########################################
    # General reconfig
    ###########################################
    cfg.gpu = args.gpu

    cfg.arch = args.arch
    cfg.DATA_TYPE = args.datatype
    cfg.batch_size = args.batch_size
    cfg.test_batch = args.test_batch
    cfg.vis_batch = args.vis_batch
    cfg.analyse_batch = args.analyse_batch
    cfg.WORK_MODE = args.work_mode
    cfg.resume_epoch = args.resume_epoch
    cfg.output_dir_name = args.output_name
    cfg.seed = args.seed

    # model specs
    cfg.num_slots = args.num_slots
    cfg.temperature = args.temperature
    cfg.latent_dim = args.latent_dim
    cfg.view_dim = args.view_dim
    cfg.min_sample_views = args.min_sample_views
    cfg.max_sample_views = args.max_sample_views
    cfg.num_vq_show = args.num_vq_show
    cfg.num_mc_samples = args.num_mc_samples
    cfg.pixel_sigma = args.pixel_sigma
    cfg.elbo_weights = {
        'kl_latent': args.kl_latent,
        'kl_spatial': args.kl_spatial,
        'exp_attention': args.exp_attention,
        'exp_nll': args.exp_nll,
        'query_nll': args.query_nll
    }

    # I/O path configurations
    cfg.DATA_ROOT = args.input_dir
    cfg.ckpt_base = args.output_dir

    # demo specs
    cfg.use_bg = args.use_bg
    cfg.traverse_v = args.traverse_v
    cfg.traverse_z = args.traverse_z
    running_cfg(cfg)

    # ---------- generating demos ----------
    run_demo(cfg)

    print("\n== Demos generated!")
    print("----------------------------------")
    print("== Check them in:  '{}' \n".format(cfg.output_dir))


##############################################################################
if __name__ == "__main__":
    cfg = CONFIG()
    main(cfg)
