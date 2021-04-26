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
# import pdb


# ------------------------- respecify important flags ------------------------
def running_cfg(cfg):
    ###########################################
    # Config i/o path
    ###########################################
    if cfg.DATA_TYPE.lower() == 'gqn_jaco':
        image_size = [64, 64]
        CLASSES = ['_background_', 'jaco', 'generic']
        cfg.v_in_dim = 7
        cfg.max_sample_views = 6
        data_dir = cfg.DATA_ROOT
        assert os.path.exists(data_dir)
        # train_data_filename = os.path.join(data_dir, 'gqn_jaco', 'gqn_jaco_train.h5')
        test_data_filename = os.path.join(data_dir, 'gqn_jaco', 'gqn_jaco_test.h5')
        # assert os.path.isfile(train_data_filename)
        assert os.path.isfile(test_data_filename)
    elif cfg.DATA_TYPE.lower() == 'clevr_mv':
        image_size = [64, 64]
        CLASSES = ['_background_', 'cube', 'sphere', 'cylinder']
        cfg.v_in_dim = 3
        cfg.max_sample_views = 6
        data_dir = cfg.DATA_ROOT
        assert os.path.exists(data_dir)
        # train_data_filename = os.path.join(data_dir, 'clevr_mv', 'clevr_mv_train.json')
        test_data_filename = os.path.join(data_dir, 'clevr_mv', 'clevr_mv_test.json')
        # assert os.path.isfile(train_data_filename)
        assert os.path.isfile(test_data_filename)
    elif cfg.DATA_TYPE.lower() == 'clevr_aug':
        image_size = [64, 64]
        CLASSES = ['_background_', 'diamond', 'duck', 'mug', 'horse', 'dolphin']
        cfg.v_in_dim = 3
        cfg.max_sample_views = 6
        data_dir = cfg.DATA_ROOT
        assert os.path.exists(data_dir)
        # train_data_filename = os.path.join(data_dir, 'clevr_aug', 'clevr_aug_train.json')
        test_data_filename = os.path.join(data_dir, 'clevr_aug', 'clevr_aug_test.json')
        # assert os.path.isfile(train_data_filename)
        assert os.path.isfile(test_data_filename)
    # ------------------- For your customised CLEVR -----------------------
    elif cfg.DATA_TYPE.lower() == 'your-clevr':
        image_size = [64, 64]
        CLASSES = ['_background_', 'xxx']
        cfg.v_in_dim = 3
        cfg.max_sample_views = 6
        data_dir = cfg.DATA_ROOT
        assert os.path.exists(data_dir)
        # train_data_filename = os.path.join(data_dir, 'your-clevr', 'your-clevr_train.json')
        test_data_filename = os.path.join(data_dir, 'your-clevr', 'your-clevr_test.json')
        # assert os.path.isfile(train_data_filename)
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
    generated_dir = os.path.join(check_dir, 'generated')
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
def run_evaluation(CFG):
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

    if 'gqn' in CFG.DATA_TYPE.lower():
        from data_loader.getGqnH5 import DataLoader
    elif 'clevr' in CFG.DATA_TYPE.lower():
        from data_loader.getClevrMV import DataLoader
    else:
        raise NotImplementedError

    eval_dataloader = DataLoader(CFG.DATA_ROOT,
                                 CFG.test_data_filename,
                                 batch_size=CFG.batch_size,
                                 shuffle=True,
                                 use_bg=CFG.use_bg)

    if 'gqn' not in CFG.DATA_TYPE.lower():
        scene_meta_info = utils.read_json(CFG.test_data_filename)['scenes']

    vis_eval_dir = CFG.generated_dir
    if not os.path.exists(vis_eval_dir):
        os.mkdir(vis_eval_dir)

    # --- dict that stores all the evaluation results ---
    EVAL_RESULT = AttrDict()
    obs_rec_record = []
    qry_obs_record = []
    obs_seg_miou = []
    qry_seg_miou = []
    GT_latents = []

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
                                     vis_train=False,
                                     vis_uncertainty=False)

        B = len(images)
        V = targets[0]['view_points'].shape[0]
        num_obs = CFG.num_vq_show

        # ----- viewpoints (ids) to be evaluated at -----
        # use these lists on GT only, as the output variables are specified by these indices already
        obs_view_ids, qry_view_ids = test_out['obs_views'], test_out['query_views']
        num_qry = len(qry_view_ids)
        assert num_obs==len(obs_view_ids)

        # ----- Task performance -----
        if CFG.eval_recon:
            x_obs, x_rec = test_out['x_images'][:, :num_obs], test_out['x_recon'][:, :num_obs]
            rmse_out = np.sqrt(np.sum((x_obs - x_rec)**2, axis=-3).reshape([B, -1]).mean(-1))
            obs_rec_record.append(rmse_out)

        if CFG.eval_qry_obs:
            xq_gt, xq_pred = test_out['x_images'][:, num_obs:], test_out['x_recon'][:, num_obs:]
            rmse_out = np.sqrt(np.sum((xq_gt - xq_pred) ** 2, axis=-3).reshape([B, -1]).mean(-1))
            qry_obs_record.append(rmse_out)

        if CFG.eval_seg:
            masks_gt = torch.stack([tar['masks'].squeeze(1) for tar in targets], dim=0).permute(0, 1, 4, 2, 3)
            m_gt, m_pred = masks_gt[:, obs_view_ids], \
                                torch.from_numpy(test_out['hiers'][:, :num_obs]).to(masks_gt.device)
            num_comps = np.asarray(list([tar['num_comps'].item()] * num_obs for tar in targets), dtype='uint8')

            # Matching: find the best pred-gt object pairs as we don't enforce any permutation in the predictions
            _, match_list = utils.match_or_compute_segmentation_iou(m_pred, m_gt, num_comps,
                                                                    threshold=1.0)
            # compute mIoU using the match we find
            obs_seg_miou += utils.match_or_compute_segmentation_iou(m_pred, m_gt, num_comps,
                                                                    match_list=match_list, threshold=1.0)[0]

        if CFG.eval_qry_seg:
            masks_gt = torch.stack([tar['masks'].squeeze(1) for tar in targets], dim=0).permute(0, 1, 4, 2, 3)
            m_gt, m_pred = masks_gt[:, qry_view_ids], \
                           torch.from_numpy(test_out['hiers'][:, num_obs:]).to(masks_gt.device)
            num_comps = np.asarray(list([tar['num_comps'].item()] * num_qry for tar in targets), dtype='uint8')

            # Matching: find the best pred-gt object pairs as we don't enforce any permutation in the predictions
            _, match_list = utils.match_or_compute_segmentation_iou(m_pred, m_gt, num_comps,
                                                                    threshold=1.0)
            # compute mIoU using the match we find
            qry_seg_miou += utils.match_or_compute_segmentation_iou(m_pred, m_gt, num_comps,
                                                                    match_list=match_list, threshold=1.0)[0]

        # ----- Save latents for disentanglement analysis -----
        if batch_id < CFG.analyse_batch and CFG.eval_dist:
            # find matches
            assert num_obs == V, "Use all available observations for disentanglement evaluation"
            assert CFG.batch_size == 1, "Must set batch size to 1 for disentanglement evaluation"
            assert "clevr" in CFG.DATA_TYPE.lower()
            masks_gt = torch.stack([tar['masks'].squeeze(1) for tar in targets], dim=0).permute(0, 1, 4, 2, 3)
            m_gt, m_pred = masks_gt[:, obs_view_ids], \
                           torch.from_numpy(test_out['hiers'][:, :num_obs]).to(masks_gt.device)
            num_comps = np.asarray(list([tar['num_comps'].item()] * num_obs for tar in targets), dtype='uint8')

            # Matching: find the best pred-gt object pairs as we don't enforce any permutation in the predictions
            _, match_list = utils.match_or_compute_segmentation_iou(m_pred, m_gt, num_comps,
                                                                    threshold=1.0)

            z_2d = test_out['2d_latents']
            z_3d = test_out['3d_latents']
            scene_meta_info = utils.read_json(CFG.test_data_filename)['scenes']
            # we need remove background reps as it is less important
            z_2d = list(z_2d[0, vid, match_list[vid][1:]] for vid in range(V))
            z_3d = list(z_3d[0, vid, match_list[vid][1:]] for vid in range(V))  # delete background
            g_latent = utils.save_latents_for_eval(z_v_out=z_2d,
                                                   z_out=z_3d,
                                                   scn_indices=test_out['scene_indices'],
                                                   qry_views=obs_view_ids,
                                                   gt_scenes_meta=scene_meta_info,
                                                   out_dir=os.path.join(CFG.output_dir, 'latents'),
                                                   save_count=count_total_samples)
            GT_latents += g_latent
        count_total_samples += len(images)

    if CFG.eval_recon:
        rmse_record = np.concatenate(obs_rec_record, axis=0).mean()
        EVAL_RESULT['rmse_rec'] = rmse_record.item()

    if CFG.eval_qry_obs:
        qry_obs_record = np.concatenate(qry_obs_record, axis=0).mean()
        EVAL_RESULT['rmse_qry'] = qry_obs_record.item()

    if CFG.eval_seg:
        EVAL_RESULT['miou_seg'] = np.stack(obs_seg_miou, axis=0).mean().item()

    if CFG.eval_qry_seg:
        EVAL_RESULT['miou_qry'] = np.stack(qry_seg_miou, axis=0).mean().item()

    if len(GT_latents) > 0:
        utils.write_json({'scene_meta': GT_latents}, os.path.join(CFG.output_dir, 'gt_latent_meta.json'))

    utils.write_json(EVAL_RESULT, os.path.join(CFG.check_dir, 'eval_{}.json'.format(CFG.DATA_TYPE)))

    return EVAL_RESULT


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
                        help='save and analyse only the first [N] batch latent codes')
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
    parser.add_argument("--eval_all", default=False, help="evaluate model with all the metrics",
                        action="store_true")
    parser.add_argument("--eval_recon", default=False, help="perform reconstruction evaluation",
                        action="store_true")
    parser.add_argument("--eval_seg", default=False, help="perform segmentation evaluation",
                        action="store_true")
    parser.add_argument("--eval_qry_obs", default=False, help="perform queried Obs.Pred. evaluation",
                        action="store_true")
    parser.add_argument("--eval_qry_seg", default=False, help="perform queried Seg.Pred. evaluation",
                        action="store_true")
    parser.add_argument("--eval_dist", default=False, help="perform disentanglement evaluation",
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

    # eval specs
    cfg.use_bg = args.use_bg

    if args.eval_all:
        cfg.eval_recon = True
        cfg.eval_seg = True
        cfg.eval_qry_obs = True
        cfg.eval_qry_seg = True
    else:
        cfg.eval_recon = args.eval_recon
        cfg.eval_seg = args.eval_seg
        cfg.eval_qry_obs = args.eval_qry_obs
        cfg.eval_qry_seg = args.eval_qry_seg

    if args.eval_dist:
        cfg.eval_dist = True

    if 'gqn' in cfg.arch.lower():
        cfg.eval_seg = False

    running_cfg(cfg)

    # ---------- RUNNING EVALUATION ----------
    eval_scores = run_evaluation(cfg)

    print("\n =========== Model '{}' Evaluated on '{}' dataset =========== \n".format(cfg.arch, cfg.DATA_TYPE))

    # print evaluation form
    if args.eval_all:
        # Recomposition
        print('\n  <Reconstruction>:')
        print('   -Rec_RMSE:    {}'.format(eval_scores['rmse_rec']))

        # Observation querying
        print('\n  <Querying observation>:')
        print('   -Qry_RMSE:    {}'.format(eval_scores['rmse_qry']))

        # Segmentation
        print('\n  <Segmentation>:')
        print('   -Seg_mIoU:     {}'.format(eval_scores['miou_seg']))

        # Segmentation querying
        print('\n  <Querying segmentation>:')
        print('   -Query_mIoU:   {}'.format(eval_scores['miou_qry']))
    else:
        if cfg.eval_recon:
            print('\n  <Reconstruction>:')
            print('   -Rec_RMSE:    {}'.format(eval_scores['rmse_rec']))
        if cfg.eval_qry_obs:
            print('\n  <Querying observation>:')
            print('   -Query_RMSE:    {}'.format(eval_scores['rmse_qry']))
        if cfg.eval_seg:
            print('\n  <Segmentation>:')
            print('   -Seg_mIoU:     {}'.format(eval_scores['miou_seg']))
        if cfg.eval_qry_seg:
            print('\n  <Querying segmentation>:')
            print('   -Query_mIoU:     {}'.format(eval_scores['miou_qry']))

    if args.eval_dist:
        print("  check the saved latent codes here:\n     {}".format(os.path.join(cfg.output_dir, 'latents')))
        print("  configure https://github.com/cianeastwood/qedr.git for disentanglement quantification.")

    print('\n ===============================================')
    print('  EVALUATION FINISHED\n\n')


##############################################################################
if __name__ == "__main__":
    cfg = CONFIG()
    main(cfg)
