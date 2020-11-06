# -*- coding: utf-8 -*-
import os


class CONFIG(object):
    # ------------------------- Device config -----------------------
    USE_CUDA = False
    use_gpu = None

    # ------------------------ General config -----------------------
    arch = 'MulMON'

    # ---------------------- Path configuration ---------------------
    # Path configuration
    #   -- DATA_DIR:      root data directory
    #   -- ckpt_base:     checkpoint saved to
    #   -- resume_path:   path to saved model (=None when training from scratch)
    DATA_ROOT = os.path.abspath('../data/MultiObj/')
    ckpt_base = os.path.abspath("./logs/")
    resume_path = None
    resume_epoch = None

    # ---------------------- Data management ---------------------
    image_size = [64, 64]  # [width, height]
    max_num_objects = 7
    num_perm_iters = 10
    # validation proportion
    validation_split = 0.167

    # Preprocessing (normalisation) configuration
    augmentation = False

    # ---------------------- Model configuration ---------------------
    # Model specification
    num_slots = 7
    pixel_sigma = 0.1
    bg_sigma = 0.09
    fg_sigma = 0.11
    temperature = 1
    v_in_dim = 3
    view_dim = 5
    latent_dim = 16
    latent_mask_size = [16, 16]
    compose_mode = 'softmax'
    WORK_MODE = 'training'  # one of ['training', 'testing']

    # validation control
    eval_classification = False
    save_val_results = True
    max_validate_shapes = None
    num_mc_samples = 20

    # Training hypers configuration
    seed = None  # an integer
    num_epochs = 1
    batch_size = 32
    lr_rate = 1e-4
    momentum = 0.9
    weight_decay = 0.0
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    optimiser = 'ADAM'  # ['ADAM', 'SGD']

    # trainging recording
    log_period = 1  # print training status on terminal (unit: step)
    val_period = 10  # validate every <#> epochs
    show_period = 10  # visualise trainign outputs every <#> epochs
    save_period = 10  # save training weights every <#> epochs

    # training a generalised ELBO:
    elbo_weights = {
        'exp_nll': 1.0,
        'kl_latent': 1.0,
    }

    # Validation configuration
    if WORK_MODE == 'training':
        resume_epoch = None
    elif WORK_MODE == 'testing':
        resume_epoch = num_epochs
    else:
        pass














