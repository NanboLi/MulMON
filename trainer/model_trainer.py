# -*- coding: utf-8 -*-
import os
import sys
import math
import torch
from trainer.base_trainer import BaseTrainer
from utils import inf_loop
import trainer._utils as utils


class ModelTrainer(BaseTrainer):
    """ Trainer (single-gpu) """

    def __init__(self, model, loss, metrics, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, step_per_epoch=None, device=None):
        super().__init__(model, loss, metrics, optimizer, config, device)
        self.config = config
        self.train_loader = train_data_loader
        if step_per_epoch is None:
            # epoch-based training
            self.step_per_epoch = len(self.train_loader)
        elif step_per_epoch:
            # iteration-based training
            self.train_loader = inf_loop(self.train_loader)  # Reusable iterator
            self.step_per_epoch = step_per_epoch

        self.do_validation = valid_data_loader is not None
        if self.do_validation:
            self.valid_loader = valid_data_loader

        self.save_period = config.save_period
        self.lr_scheduler = lr_scheduler

        if self.config.resume_epoch is not None:
            self.start_epoch = self.config.resume_epoch + 1
            if self.lr_scheduler is not None:
                self.lr_scheduler.last_epoch = self.config.resume_epoch * self.step_per_epoch
                self.optimizer.param_groups[0]['lr'] = self.lr_scheduler.get_lr()[0]

        # self.save_dir has been declared in parenthese
        assert config.vis_train_dir is not None
        self.vis_train_dir = config.vis_train_dir
        assert config.generated_dir is not None
        self.generated_dir = config.generated_dir

    def train(self):
        print('\n================== Start training ===================')
        # self.start_time = time.time()
        assert (self.epochs + 1) > self.start_epoch
        for epoch in range(self.start_epoch, self.epochs + 1):
            print("Epoch {}".format(epoch))
            self._train_epoch(epoch)

            if epoch % self.config.val_period == 0:
                valid_outs = self._validate_epoch(epoch)

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

        print('models have been saved to {}'.format(self.save_dir))
        print('================= Training finished =================\n')

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        iter_count = (epoch - 1) * self.step_per_epoch

        for bid, (images, targets) in enumerate(
                metric_logger.log_every(self.train_loader, self.config.log_period, len_iter=self.step_per_epoch,
                                        header=header)):
            images = list(image.to(self.device).detach() for image in images)
            targets = [{k: v.to(self.device).detach() for k, v in t.items()} for t in targets]

            if self.config.temperature != 0.0:
                std_global = max(self.config.temperature * (1.0 - float(iter_count + bid) / 20000.0),
                                 self.config.pixel_sigma)
            else:
                std_global = self.config.pixel_sigma

            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets, std=std_global)
            losses = sum(loss for loss in loss_dict.values())

            # --- back prop gradients ---
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)

            # logging ins
            if self.writer is not None:
                for kwd in loss_dict_reduced.keys():
                    self.writer.add_scalar('train/{}'.format(kwd), loss_dict_reduced[kwd], iter_count + bid)
                self.writer.add_scalar('std_annealing', std_global, iter_count + bid)

            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                if self.writer is not None:
                    self.writer.add_scalar('lr/optimizer', self.lr_scheduler.get_lr()[0], iter_count + bid)

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

            if bid == self.step_per_epoch:
                break

    def _validate_epoch(self, epoch):
        """
        Validate model on validation data and save visual results for checking
        :return: a dict of model's output
        """
        self.model.eval()
        if epoch % self.config.show_period == 0:
            vis_epo_dir = os.path.join(self.vis_train_dir, 'epoch_{}'.format(epoch))
            if not os.path.exists(vis_epo_dir):
                os.mkdir(vis_epo_dir)
        else:
            vis_epo_dir = None
        with torch.no_grad():
            images, targets = next(iter(self.valid_loader))
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            val_outs = self.model.predict(images, targets, save_sample_to=vis_epo_dir)
        return val_outs

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        """
        d = {
             'epoch': epoch,
             'model': self.model.state_dict(),
             'optimizer': self.optimizer.state_dict()
        }
        filename = os.path.abspath(os.path.join(self.save_dir,
                                                'checkpoint-epoch{}.pth'.format(epoch)))
        torch.save(d, filename)

    def _resume_checkpoint(self, resume_path, optimizer=None):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        :param optimizer: Specify whether using a new optimizer if provided or stick with the previous
        """
        ckpt = torch.load(resume_path)
        self.model.load_state_dict(ckpt['model'], strict=True)
        self.start_epoch = ckpt['epoch']
        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
