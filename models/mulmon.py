import os
import random
import numpy as np
from attrdict import AttrDict
import torch
import torch.nn as nn
import torch.distributions as dist
from models.modules import *
import visualisation as vis
import utils


class MulMON(nn.Module):
    """
    v1: object-centric GQN.
       a. learning scenes from T available multi-view observations.
       b. querying scene synthesis and segmentation
    """
    def __init__(self,
                 config,
                 name='MulMON'):
        super(MulMON, self).__init__()
        self.name = name
        self.config = config
        self.training = True if self.config.WORK_MODE == 'training' else False
        self.nit_innerloop = 5  # num_iterations of the inner loop
        self.K = self.config.num_slots
        self.v_in_dim = self.config.v_in_dim
        self.z_dim = self.config.latent_dim
        self.v_dim = self.config.view_dim
        # self.v_dim = self.z_dim
        self.std = self.config.pixel_sigma
        self.min_num_views = self.config.min_sample_views
        self.max_num_views = self.config.max_sample_views
        self.num_vq_show = self.config.num_vq_show
        self.decoder = SpatialBroadcastDec(self.z_dim, 4, self.config.image_size)
        self.refine_net = RefineNetLSTM(self.z_dim, channels_in=17, image_size=self.config.image_size)
        self.view_encoder = nn.Sequential(
            nn.Linear(self.v_in_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.v_dim, bias=True)
        )
        self.projector = nn.Sequential(
            nn.Linear(self.z_dim + self.v_dim, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.z_dim, bias=True)
        )
        self.lmbda0 = nn.Parameter(torch.randn(1, 2*self.z_dim) - 0.5, requires_grad=True)

    @staticmethod
    def save_visuals(vis_images, vis_recons, vis_comps, vis_hiers, save_dir, start_id=0):
        """
        Used for visualising training procedure as well as the prediction of new input
        :param vis_images: (a list of [B, V, 3, H, W] tensors) the given scene images for learning.
        :param vis_recons: (a tensor [B, V, 3, H, W]) the reconstructed scene images.
        :param vis_comps: (a tensor [B, V, K, 3, H, W]) the generated scene-component appearances.
        :param vis_hiers: (a tensor [B, V, K, 1, H, W]) the occlusion hierarchy (visibility), i.e. mixing coefficients.
        :param save_dir: (optional) if specified, then the generate sample images will be saved in the very dir.
        :param start_id: (optional) global indicator of filenames, useful to save multiple batches.
        """
        B, V, K, _, H, W = vis_comps.shape
        for count in range(B):
            ST = []
            for v in range(V):
                # save the queried target images
                img = vis_images[count, v, ...].transpose([1, 2, 0])
                ST.append(img)

                # save the reconstructed image
                rec = vis_recons[count, v, ...].transpose([1, 2, 0])
                ST.append(rec)

                # save and check components
                for k in range(K):
                    c_app = vis_comps[count, v, k, ...].transpose([1, 2, 0])
                    ST.append(c_app)

                # save the hierarchy as depth-order maps
                d_order = np.argmax(vis_hiers[count, v, ...], axis=0)
                d_order = d_order.astype('uint8')
                ST.append(vis.save_dorder_plots(d_order, K_comps=K))
            vis.save_images_grid(ST, nrows=3+K, ncols=V, fig_size=8,
                                 save_to=os.path.join(save_dir, '{:06d}.png'.format(count + start_id)))
            print(" -- {} generated scene samples have been saved".format(count + start_id + 1))

    @staticmethod
    def save_visuals_eval(num_obs_views, vis_images, vis_recons, vis_comps, vis_hiers, save_dir, start_id=0):
        B, V, K, _, H, W = vis_comps.shape
        for b in range(B):
            prefix = 'obs'
            for v in range(V):
                if v >= num_obs_views:
                    prefix = 'qry'

                # --- save the observed --- #
                vis.save_single_image(vis_images[b, v, ...].transpose([1, 2, 0]),
                                      os.path.join(save_dir, '{}_{}{}_alpha.png'.format(start_id + b, prefix, v)))

                # --- save the generated ---#
                vis.save_single_image(vis_recons[b, v, ...].transpose([1, 2, 0]),
                                      os.path.join(save_dir, '{}_{}{}_beta.png'.format(start_id + b, prefix, v)))
                for k in range(K):
                    vis.save_single_image(vis_comps[b, v, k, ...].transpose([1, 2, 0]),
                                          os.path.join(save_dir, '{}_{}{}_c{}.png'.format(start_id + b, prefix, v, k)))

                # --- save the spatial reasoning --- #
                # hiers: [B, V, K, H, W]
                seg = np.argmax(vis_hiers[b, v, ...], axis=0).astype('uint8')
                vis_seg = vis.save_dorder_plots(seg, K_comps=K)
                vis.save_single_image(vis_seg,
                                      os.path.join(save_dir, '{}_{}{}_seg.png'.format(start_id + b, prefix, v)))
            print(" -- {} generated scene samples have been saved".format(b + start_id + 1))

    @staticmethod
    def sample_view_config(num_views, min_limit=5, max_limit=7, allow_repeat=False):
        """
        Select the views the scenes are observed from, also generate viewpoints for the querying
        :param num_views: (default depends on the data, e.g. 10) how many viewpoints are available, or reachable for the
                            agents
        :param min_limit: minimum number of observations one agent needs
        :param max_limit: maximum number of observations the agent are provided
        :param allow_repeat: allow repeated viewpoints to be sampled
        :return: observation_view_id_list, querying_view_id_list
        """
        assert max_limit <= num_views
        FULL_HOUSE = [*range(num_views)]
        # randomise the order of views
        random.shuffle(FULL_HOUSE)
        # randomise the number of the given observations
        L = random.randint(min_limit, max_limit)

        if L == num_views:
            return FULL_HOUSE, FULL_HOUSE

        # random partition of the observation views and query views
        observation_view_id_list = FULL_HOUSE[:L]
        if allow_repeat:
            querying_view_id_list = random.sample(FULL_HOUSE, num_views-L)
        else:
            querying_view_id_list = FULL_HOUSE[L:]
        return observation_view_id_list, querying_view_id_list

    def get_refine_inputs(self, _x, mu_x, masks, mask_logits, ll_pxl, lmbda, loss, ll_col):
        """
        Generate inputs to refinement network
        (adapted from: https://github.com/MichaelKevinKelly/IODINE)
        """
        N, K, C, H, W = mu_x.shape

        # Calculate additional non-gradient inputs
        # ll_pxl is the mixture log-likelihood [N, 1, 3, H, W]
        # ll_col is the color gaussian ll, not account for masks [N, K, 3, H, W]
        x_lik = ll_pxl.sum(dim=2, keepdim=True).exp().repeat(1, K, 1, 1, 1)  # [N, K, 1, H, W]
        col_lik = torch.softmax(ll_col.sum(2, keepdim=True), dim=1)  # (N,K,1,H,W)

        # This computation is a little weird. Basically we do not count one of the slot.
        K_likelihood = ll_col.sum(dim=2, keepdim=True).exp()  # (B, K, 1, H, W)
        # likelihood = (B, 1, 1, H, W), self.mask (B, K, 1, H, W)
        likelihood = (masks * K_likelihood).sum(dim=1, keepdim=True)  # (B, 1, 1, H, W)
        # leave_one_out (B, K, 1, H, W)
        leave_one_out = likelihood - masks * K_likelihood
        # normalize
        leave_one_out = leave_one_out / (1 - masks + 1e-5)

        # Calculate gradient inputs
        dmu_x = torch.autograd.grad(loss, mu_x, retain_graph=True, only_inputs=True)[0]  ## (N,K,C,H,W)
        dmasks = torch.autograd.grad(loss, masks, retain_graph=True, only_inputs=True)[0]  ## (N,K,1,H,W)
        dlmbda = torch.autograd.grad(loss, lmbda, retain_graph=True, only_inputs=True)[0]  ## (N*K,2*z_dim)

        # Layer norm -- stablises trainings
        x_lik_stable = layernorm(x_lik).detach()
        leave_one_out_stable = layernorm(leave_one_out).detach()
        dmu_x_stable = layernorm(dmu_x).detach()
        dmasks_stable = layernorm(dmasks).detach()
        dlmbda_stable = layernorm(torch.stack(dlmbda.chunk(N, 0), 0)).detach()
        dlmbda_stable = torch.cat(dlmbda_stable.split(1, dim=0), dim=1).squeeze(0)

        # Generate coordinate channels
        xx = torch.linspace(-1, 1, W, device=_x.device)
        yy = torch.linspace(-1, 1, H, device=_x.device)
        yy, xx = torch.meshgrid((yy, xx))
        # (2, H, W)
        coords = torch.stack((xx, yy), dim=0)
        coords = coords[None, None].repeat(N, self.K, 1, 1, 1).detach()

        # Concatenate into vec and mat inputs
        img_args = (_x, mu_x, masks, mask_logits, dmu_x_stable, dmasks_stable,
                    col_lik, x_lik_stable, leave_one_out_stable, coords)
        state_args = (lmbda, dlmbda_stable)
        img_inp = torch.cat(img_args, dim=2)
        state_inp = torch.cat(state_args, dim=1)

        # Reshape
        img_inp = img_inp.view((N * K,) + img_inp.shape[2:])

        return {'img': img_inp, 'state': state_inp}

    def decode(self, z):
        K = self.K
        B = z.size(0) // K
        # Get means and masks
        dec_out = self.decoder(z)  # (B*K,4,H,W) where 4= 3(RGB) + 1(alpha)
        mu_x, mask_logits = dec_out[:, :3, :, :], dec_out[:, 3:, :, :]
        mask_logits = mask_logits.reshape((B, K,) + mask_logits.shape[1:])
        mu_x = mu_x.reshape((B, K,) + mu_x.shape[1:])
        return mask_logits, mu_x

    def _iterative_inference(self, x, y, lmbda, niter=1):
        """
        :param x: [B, 3, H, W]
        :param lmbda:  [B*K, 2*D]
        """
        B, C, _, _ = x.size()
        K = self.K
        total_loss = 0.

        # Initialize LSTMCell hidden states
        h = torch.zeros(B * K, 128, device=x.device, requires_grad=True)
        c = torch.zeros_like(h, device=x.device, requires_grad=True)
        assert h.max().item() == 0. and h.min().item() == 0.
        mu_pri, logvar_pri = lmbda.chunk(2, dim=1)

        balancing_discount = 1.0 / (0.5 * (niter + 1.0))
        for it in range(niter):
            # Sample latent code
            mu_z, logvar_z = lmbda.chunk(2, dim=1)
            z = dist.Normal(mu_z, to_sigma(logvar_z)).rsample()  # (N*K,z_dim)
            kl_qz = kl_exponential(mu_z, to_sigma(logvar_z),
                                        pri_mu=mu_pri, pri_sigma=to_sigma(logvar_pri), z_samples=z)
            kl_qz = torch.stack(kl_qz.chunk(B, dim=0), dim=0).sum(dim=(1, 2))

            # Obtain view-conditioned scene representations
            z_y = self.projector(torch.cat((z, y.reshape(B*K, -1)), dim=-1))

            # Generate independent components (object 2d geometries + RGB values)
            mask_logits, mu_x = self.decode(z_y)

            # get the mixing coefficients (Categorical parameters)
            masks = torch.softmax(mask_logits, dim=1)  # (N,K,1,H,W)

            # Compute the loss (neg ELBO): reconstruction (nll) & KL divergence
            _x = x.unsqueeze(dim=1).repeat(1, K, 1, 1, 1)
            ll_pxl, ll_col = Gaussian_ll(mu_x, _x, masks, self.std)  # (N,1,3,H,W)
            nll = -1. * (ll_pxl.flatten(start_dim=1).sum(dim=-1).mean()) * self.config.elbo_weights['exp_nll']
            loss = nll + kl_qz.mean() * self.config.elbo_weights['kl_latent']

            # Accumulate loss
            total_loss = total_loss + loss * ((float(it) + 1) / float(niter))

            assert not torch.isnan(loss).any().item(), 'Loss at t={} is nan. (nll,div): ({},{})'.format(nll, kl_qz)
            if it == niter - 1:
                continue

            # Refine lambda
            refine_inp = self.get_refine_inputs(_x, mu_x, masks, mask_logits, ll_pxl, lmbda, loss, ll_col)
            delta, h, c = self.refine_net(refine_inp, h, c)
            assert not torch.isnan(lmbda).any().item(), 'Lmbda at t={} has nan: {}'.format(it, lmbda)
            assert not torch.isnan(delta).any().item(), 'Delta at t={} has nan: {}'.format(it, delta)
            lmbda = lmbda + delta
            assert not torch.isnan(lmbda).any().item(), 'Lmbda at t={} has nan: {}'.format(it, lmbda)

        return balancing_discount*total_loss, lmbda, mask_logits, mu_x

    def sample_qz_uncertainty(self, lmbda, yq):
        """image-space pixel-wise uncertainty estimation via MC sampling"""
        B, K, _ = yq.size()
        mu_z, logvar_z = lmbda.chunk(2, dim=1)
        z = dist.Normal(mu_z, to_sigma(logvar_z)).rsample([self.config.num_mc_samples])
        var_x = []
        # var_obj = []
        for s in range(self.config.num_mc_samples):
            z_yq = self.projector(torch.cat((z[s], yq.reshape(B * K, -1)), dim=-1))
            mask_logits, mu_x = self.decode(z_yq)
            var_x.append(utils.numpify(torch.sum(torch.softmax(mask_logits, dim=1) * mu_x, dim=1)))  # [B, 3, H, w]
            # var_obj.append(utils.numpify(torch.sigmoid(mask_logits) * mu_x))  # [B, K, 3, H, w]
        var_x = np.stack(var_x, axis=0).var(axis=0, ddof=1).sum(1)
        # var_obj = np.stack(var_obj, axis=0).var(axis=0, ddof=1).sum(2)
        return var_x, None    # var_obj

    def forward(self, images, targets, std=None):
        xmul = torch.stack(images, dim=0)  # [B, V, C, H, W]
        v_pts = torch.stack([tar['view_points'] for tar in targets], dim=0).type(xmul.dtype)  # [B, V, 3]
        # adding noise to viewpoint vectors helps to robustify the model:
        v_pts += 0.015*torch.randn_like(v_pts, dtype=xmul.dtype, device=xmul.device, requires_grad=False)

        B, V, _, _, _ = xmul.size()
        K, nit_inner_loop, z_dim = self.K, self.nit_innerloop, self.z_dim

        if std:
            self.std = std

        # Random partition of observation viewpoints and query viewpoints
        obs_view_idx, qry_view_idx = self.sample_view_config(V, self.min_num_views, self.max_num_views,
                                                             allow_repeat='gqn' in self.config.DATA_TYPE)

        # Initialize parameters for the latents' distribution
        assert not torch.isnan(self.lmbda0).any().item(), 'lmbda0 has nan'
        lmbda = self.lmbda0.expand((B * K,) + self.lmbda0.shape[1:])
        neg_elbo = 0.   # torch.tensor(B, 1, device=xmul.device, requires_grad=True)
        xq_nll = 0.  # torch.zeros(B, 1, device=xmul.device, requires_grad=True)

        # --- get view codes ---
        v_feat = self.view_encoder(v_pts.reshape(B * V, -1))  # output [B*V, 8]
        v_feat = v_feat.reshape(B, V, -1).unsqueeze(1).repeat(1, K, 1, 1)

        # --- scene learning phase --- #
        discount_obs = 1.0 / (0.5 * (len(obs_view_idx) + 1.0))
        # Outer loop: learning scenes from multiple views
        for venum, v in enumerate(obs_view_idx):
            x = xmul[:, v, ...]
            y = v_feat[:, :, v, :]

            # Inner loop: single-view scene learning in an iterative fashion
            nelbo_v, lmbda, _, _ = self._iterative_inference(x, y, lmbda, nit_inner_loop)
            neg_elbo = neg_elbo + nelbo_v.mean() * ((float(venum) + 1) / float(len(obs_view_idx)))

        # --- scene querying phase --- #
        for vqnum, vq in enumerate(qry_view_idx):
            x = xmul[:, vq, ...]
            yq = v_feat[:, :, vq, :]

            # Sample 3D-informative object latents
            mu_z, logvar_z = lmbda.chunk(2, dim=1)
            z = dist.Normal(mu_z, to_sigma(logvar_z)).rsample()

            # Project the 3D latents to 2D w.r.t
            z_yq = self.projector(torch.cat((z, yq.reshape(B * K, -1)), dim=-1))

            mask_logits, mu_x = self.decode(z_yq)
            # get masks
            masks = torch.softmax(mask_logits, dim=1)
            ll_pxl, _ = Gaussian_ll(mu_x, x.unsqueeze(dim=1).expand((B, K,) + x.shape[1:]),
                                         masks, self.std)  # (N,1,3,H,W)
            nll = -1. * (ll_pxl.flatten(start_dim=1).sum(dim=-1).mean())

            xq_nll = xq_nll + nll.mean() * (1.0 / float(len(qry_view_idx)))

        loss_dict = AttrDict()
        loss_dict['neg_elbo'] = neg_elbo * discount_obs
        loss_dict['query_nll'] = xq_nll * self.config.elbo_weights['query_nll']
        return loss_dict

    @torch.enable_grad()
    def predict(self, images, targets,
                save_sample_to=None,
                save_start_id=0,
                vis_train=True,
                vis_uncertainty=False):
        """
        We show uncertainty
        """
        xmul = torch.stack(images, dim=0)  # [B, V, C, H, W]
        v_pts = torch.stack([tar['view_points'] for tar in targets], dim=0).type(xmul.dtype)  # [B, V, 3]

        B, V, _, _, _ = xmul.size()
        K, nit_inner_loop, z_dim = self.K, self.nit_innerloop, self.z_dim

        # sample the number of observations and which observations
        obs_view_idx, qry_view_idx = self.sample_view_config(V, self.num_vq_show, self.num_vq_show,
                                                             allow_repeat=False)

        # Initialize parameters for latents' distribution
        assert not torch.isnan(self.lmbda0).any().item(), 'lmbda0 has nan'
        lmbda = self.lmbda0.expand((B * K,) + self.lmbda0.shape[1:])

        # --- get view codes ---
        v_feat = self.view_encoder(v_pts.reshape(B * V, -1))  # output [B*V, 8]
        v_feat = v_feat.reshape(B, V, -1).unsqueeze(1).repeat(1, K, 1, 1)

        # --- record for visualisation --- #
        vis_images = []
        vis_recons = []
        vis_comps = []
        vis_hiers = []
        vis_2d_latents = []
        vis_3d_latents = []

        # --- scene learning phase --- #
        for venum, v in enumerate(obs_view_idx):
            x = xmul[:, v, ...]
            y = v_feat[:, :, v, :]

            # Knowledge summarize in an iterative fashion (does not have to be though: set T=1)
            nelbo_v, lmbda, m_logits, mu_x = self._iterative_inference(x, y, lmbda, nit_inner_loop)
            masks = torch.softmax(m_logits, dim=1)

            # get independent object silhouette
            indi_masks = torch.sigmoid(m_logits)

            vis_images.append(utils.numpify(x))
            vis_recons.append(utils.numpify(torch.sum(masks*mu_x, dim=1)))
            vis_comps.append(utils.numpify(indi_masks*mu_x))
            vis_hiers.append(utils.numpify(masks))

            del mu_x, m_logits, masks

        # --- scene querying phase --- #
        assert len(qry_view_idx) > 0
        for vqnum, vq in enumerate(qry_view_idx):
            x = xmul[:, vq, ...]
            yq = v_feat[:, :, vq, :]

            # making view-dependent generation
            mu_z, logvar_z = lmbda.chunk(2, dim=1)
            z = dist.Normal(mu_z, to_sigma(logvar_z)).rsample()

            z_yq = self.projector(torch.cat((z, yq.reshape(B * K, -1)), dim=-1))

            mask_logits, mu_x = self.decode(z_yq)
            # get masks
            masks = torch.softmax(mask_logits, dim=1)

            # get independent object silhouette
            indi_masks = torch.sigmoid(mask_logits)
            # uncomment the below to binarize the silhouette with a tunable threshold (default: 0.5).
            # indi_masks = (indi_masks > 0.5).type(mu_x.dtype)

            vis_images.append(utils.numpify(x))
            vis_recons.append(utils.numpify(torch.sum(masks * mu_x, dim=1)))
            vis_comps.append(utils.numpify(indi_masks * mu_x))
            vis_hiers.append(utils.numpify(masks))
            vis_2d_latents.append(utils.numpify(z_yq.reshape(B, K, -1)))
            vis_3d_latents.append(utils.numpify(mu_z.reshape(B, K, -1)))

            del mu_x, mask_logits, masks

        vis_images = np.stack(vis_images, axis=1)  # [B, V, 3, H, W]
        vis_recons = np.stack(vis_recons, axis=1)  # [B, V, 3, H, W]
        vis_comps = np.stack(vis_comps, axis=1)  # [B, V, K, 3, H, W]
        vis_hiers = np.squeeze(np.stack(vis_hiers, axis=1), axis=3)  # [B, V, K, H, W]
        vis_2d_latents = np.stack(vis_2d_latents, axis=1)  # [B, qV, K, D]
        vis_3d_latents = np.stack(vis_3d_latents, axis=1)  # [B, qV, K, D]

        if save_sample_to is not None:
            if vis_train:
                self.save_visuals(vis_images, vis_recons, vis_comps, vis_hiers,
                                  save_dir=save_sample_to,
                                  start_id=save_start_id)
            else:
                self.save_visuals_eval(len(obs_view_idx), vis_images, vis_recons, vis_comps, vis_hiers,
                                       save_dir=save_sample_to,
                                       start_id=save_start_id)
        preds = {
            'x_images': vis_images,
            'x_recon': vis_recons,
            'x_comps': vis_comps,
            'hiers': vis_hiers,
            'lmbda': lmbda,
            '2d_latents': vis_2d_latents,
            '3d_latents': vis_3d_latents,
            'scene_indices': list(tar['scn_id'][0].item() for tar in targets),
            'obs_views': obs_view_idx,
            'query_views': qry_view_idx
        }
        return preds

    @torch.no_grad()
    def v_travel(self, lmbda, v_pts, save_sample_to=None, save_start_id=0):
        """
        Viewpoint queired predictions along a viewpoint trajectory.

        :param z: [B*K, D]
        :param v_pts: [B, L, dview]   (a viewpoint trajectory)
        """
        save_dir = os.path.join(save_sample_to, 'v_track')
        utils.ensure_dir(save_dir)
        B, L, _ = v_pts.size()
        K = lmbda.size(0) // B

        mu_z, logvar_z = lmbda.chunk(2, dim=-1)
        z = dist.Normal(mu_z, to_sigma(logvar_z)).rsample()

        v_feat = self.view_encoder(v_pts.reshape(B * L, -1))  # output [B*V, 8]
        v_feat = v_feat.reshape(B, L, -1).unsqueeze(1).repeat(1, K, 1, 1)

        GIFs = {
            'alpha': [],    # RGB images
            'seg': [],      # Segmentation
            'uncer': []     # Unvertainty (pixel-wise space)
        }
        for l in range(L):
            yq = v_feat[:, :, l, :]
            z_yq = self.projector(torch.cat((z, yq.reshape(B*K, -1)), dim=-1))
            mask_logits, mu_x = self.decode(z_yq)
            masks = torch.softmax(mask_logits, dim=1)
            x_hat = torch.sum(masks * mu_x, dim=1)
            uncer, _ = self.sample_qz_uncertainty(lmbda, yq)
            GIFs['alpha'].append(x_hat)
            GIFs['seg'].append(masks.squeeze(2))
            GIFs['uncer'].append(torch.from_numpy(uncer).to(masks).float())

        for key in GIFs.keys():
            GIFs[key] = torch.stack(GIFs[key], dim=0)   # [steps, B, #, C, H, W]

        for b in range(B):
            save_batch_dir = os.path.join(save_dir, str(b+save_start_id))
            utils.ensure_dir(save_batch_dir)
            for key in GIFs.keys():
                prefix = '{}{}'.format(b + save_start_id, key)
                for iid in range(L):
                    if key == 'alpha':
                        vis.enhance_save_single_image(utils.numpify(GIFs[key][iid, b, ...].cpu().permute(1, 2, 0)),
                                                      os.path.join(save_batch_dir, '{}_{:02d}.png'.format(prefix, iid)))
                        # save_image(tensor=GIFs[key][iid, b, ...].cpu(),
                        #            filename=os.path.join(save_batch_dir, '{}_{:02d}.jpg'.format(prefix, iid)))
                    elif key == 'seg':
                        seg = np.argmax(utils.numpify(GIFs[key][iid, b, ...].cpu()),
                                        axis=0).astype('uint8')
                        seg = vis.save_dorder_plots(seg, K_comps=K, cmap='hsv')
                        vis.save_single_image(seg,
                                              os.path.join(save_batch_dir, '{}_{:02d}.png'.format(prefix, iid)))
                    elif key == 'uncer':
                        vis_var = np.log10(utils.numpify(GIFs[key][iid, b, ...].cpu()) + 1e-6)
                        vis_var = vis.map_val_colors(vis_var,
                                                     v_min=-6., v_max=-2., cmap='hot')
                        vis.save_single_image(vis_var,
                                              os.path.join(save_batch_dir, '{}_{:02d}.png'.format(prefix, iid)))
                    else:
                        raise NotImplementedError
                vis.grid2gif(str(os.path.join(save_batch_dir, prefix + '*.png')),
                             str(os.path.join(save_batch_dir, '{}.gif'.format(prefix))), delay=20)

    @torch.no_grad()
    def z_travel(self, z3d, v_pts, limit=3.0, int_step_size=0.66, save_sample_to=None, save_start_id=0):
        """
        Traverse latent space to visualise the learnt reps.

        :param z3d:  [B, K, D]
        :param v_pts:  [B, L, dView]
        :param limit:  numerical bounds for traverse
        :param int_step_size:  traverse step size (interpolation gap between traverse points)
        :param save_dir:  save the output to this dir
        :param start_id:  save the output as file
        """
        from torchvision.utils import save_image
        save_dir = os.path.join(save_sample_to, 'disen_3d')
        utils.ensure_dir(save_dir)
        B, K, D = z3d.size()
        V = v_pts.size(1)

        v_feat = self.view_encoder(v_pts.reshape(B * V, -1))  # output [B*V, 8]
        v_feat = v_feat.reshape(B, V, -1).unsqueeze(1).repeat(1, K, 1, 1)

        H, W = tuple(self.config.image_size)
        interpolation = torch.arange(-limit, limit + 0.1, int_step_size)

        gifs = []
        # ------------ Select intereted object and informtive latent dimensions here ------------
        k = 2  # we select only one object out of K for analysis
        # SPECIFY_DIMENSIONS=[9, 31]
        # D = len(SPECIFY_DIMENSIONS)
        # ---------------------------------------------------------------------------------------
        for d in range(D):
            for int_val in interpolation:
                z = z3d.clone()  # [B, K, D]
                z[:, k, d] += int_val

                for vq in range(V):
                    yq = v_feat[:, :, vq, :]
                    z_yq = self.projector(torch.cat((z.reshape(B * K, -1), yq.reshape(B * K, -1)), dim=-1))
                    mask_logits, mu_x = self.decode(z_yq)
                    gifs.append(torch.sum(torch.softmax(mask_logits, dim=1) * mu_x, dim=1).data)
        gifs = torch.cat(gifs, dim=0)
        gifs = gifs.reshape(D, len(interpolation), V, B, 3, H, W).permute([3, 0, 1, 2, 4, 5, 6])

        for b in range(B):
            save_batch_dir = os.path.join(save_dir, str(save_start_id + b))
            utils.ensure_dir(save_batch_dir)
            b_gifs = gifs[b, ...]
            b_gifs = torch.cat(b_gifs.chunk(V, dim=2), dim=0).squeeze(2)

            for iid in range(len(interpolation)):
                key = 'frame'
                vis.torch_save_image_enhanced(tensor=b_gifs[:, iid, ...].cpu(),
                                              filename=os.path.join(save_batch_dir, '{}_{:02d}.jpg'.format(key, iid)),
                                              nrow=D, pad_value=1, enhance=True)
            vis.grid2gif(str(os.path.join(save_batch_dir, key + '*.jpg')),
                         str(os.path.join(save_batch_dir, 'disten3d.gif')), delay=20)

            print(" -- traversed latent space for {} scene samples".format(b + 1))
