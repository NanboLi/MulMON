#!/bin/bash

# makde sure the below path are correctly configured before run the project
data_path=<YOUR-PATH>/mulmon_datasets/clevr
repo_path=.
log_path=${repo_path}/logs
data_type=clevr_mv

seed=0               # random seeds:  0 234 1994 2888 77777 ...
allow_obs=5          # (important) how many observations are available
epoch=2000
which_gpu=1

python eval.py --arch mv_MulMON --datatype ${data_type} --work_mode testing --gpu ${which_gpu} \
--input_dir ${data_path} --output_dir ${log_path} --output_name ev${epoch}_obs${allow_obs} \
--resume_epoch ${epoch} --batch_size 4 --test_batch 50 --vis_batch 0 --analyse_batch 0 --seed ${seed} \
--num_slots 7 --pixel_sigma 0.1 --latent_dim 16 --view_dim 5 --min_sample_views 1 --max_sample_views 6 --num_vq_show ${allow_obs} --num_mc_samples 20 \
--query_nll 1.0 --exp_nll 1.0 --exp_attention 1.0 --kl_latent 1.0 --kl_spatial 1.0 \
--use_bg --eval_all # --eval_dist