import os
import math, random
import numpy as np
from attrdict import AttrDict
import torch
import torch.nn as nn
import torch.distributions as dist
from models.encoder import Encoder
from models.decoder import SpatialBroadcastDec
import visualisation as vis
import utils


def to_sigma(logvar):
    """ Compute std """
    return torch.exp(0.5*logvar)


class RefineNetLSTM(nn.Module):
    """
    function Phi (see Sec 3.3 of the paper)
    (adapted from: https://github.com/MichaelKevinKelly/IODINE)
    """
    def __init__(self, z_dim, channels_in, image_size):
        super(RefineNetLSTM, self).__init__()
        self.convnet = Encoder(channels_in, 128, image_size)
        self.lstm = nn.LSTMCell(128 + 4 * z_dim, 128, bias=True)
        self.fc_out = nn.Linear(128, 2 * z_dim)

    def forward(self, x, h, c):
        x_img, lmbda_moment = x['img'], x['state']
        conv_codes = self.convnet(x_img)
        lstm_input = torch.cat((lmbda_moment, conv_codes), dim=1)
        h, c = self.lstm(lstm_input, (h, c))
        return self.fc_out(h), h, c


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
        :return: observation_view_id_list, querying_view_id_list
        """
        assert max_limit <= num_views
        FULL_HOUSE = [*range(num_views)]
        # randomise the order of views
        random.shuffle(FULL_HOUSE)
        # randomise the number of given observations
        L = random.randint(min_limit, max_limit)

        # random partition of observation views and query views
        observation_view_id_list = FULL_HOUSE[:L]
        if allow_repeat:
            querying_view_id_list = random.sample(FULL_HOUSE, num_views-L)
        else:
            querying_view_id_list = FULL_HOUSE[L:]

        return observation_view_id_list, querying_view_id_list

    @staticmethod
    def Gaussian_ll(x_col, _x, masks, std):
        """
        x_col: [B,K,C,H,W]
        _x:    [B,K,3,H,W]
        masks:   [B,K,1,H,W]
        """
        B, K, _, _, _ = x_col.size()
        std_t = torch.tensor([std] * K, device=x_col.device, dtype=x_col.dtype, requires_grad=False)
        std_t = std_t.expand(1, K).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        log_pxz = dist.Normal(x_col, std_t).log_prob(_x)
        ll_pix = torch.logsumexp((masks + 1e-6).log() + log_pxz, dim=1, keepdim=True)  # [B,K,3,H,W]
        assert ll_pix.min().item() > -math.inf
        return ll_pix, log_pxz

    @staticmethod
    def kl_exponential(post_mu, post_sigma, z_samples=None, pri_mu=None, pri_sigma=None):
        """Support Gaussian only now"""
        if pri_mu is None:
            pri_mu = torch.zeros_like(post_mu, device=post_mu.device, requires_grad=True)
        if pri_sigma is None:
            pri_sigma = torch.ones_like(post_sigma, device=post_sigma.device, requires_grad=True)
        p_post = dist.Normal(post_mu, post_sigma)
        if z_samples is None:
            z_samples = p_post.rsample()
        p_pri = dist.Normal(pri_mu, pri_sigma)
        return p_post.log_prob(z_samples) - p_pri.log_prob(z_samples)

    @staticmethod
    def layernorm(x):
        """
        :param x: (B, K, L) or (B, K, C, H, W)
        (function adapted from: https://github.com/MichaelKevinKelly/IODINE)
        """
        if len(x.size()) == 3:
            layer_mean = x.mean(dim=2, keepdim=True)
            layer_std = x.std(dim=2, keepdim=True)
        elif len(x.size()) == 5:
            mean = lambda x: x.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
            layer_mean = mean(x)
            # this is not implemented in some version of torch
            layer_std = torch.pow(x - layer_mean, 2)
            layer_std = torch.sqrt(mean(layer_std))
        else:
            assert False, 'invalid size for layernorm'

        x = (x - layer_mean) / (layer_std + 1e-5)
        return x

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
        x_lik_stable = self.layernorm(x_lik).detach()
        leave_one_out_stable = self.layernorm(leave_one_out).detach()
        dmu_x_stable = self.layernorm(dmu_x).detach()
        dmasks_stable = self.layernorm(dmasks).detach()
        dlmbda_stable = self.layernorm(torch.stack(dlmbda.chunk(N, 0), 0)).detach()
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
            kl_qz = self.kl_exponential(mu_z, to_sigma(logvar_z),
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
            ll_pxl, ll_col = self.Gaussian_ll(mu_x, _x, masks, self.std)  # (N,1,3,H,W)
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
        """image space uncertainty MC estimation"""
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
            ll_pxl, _ = self.Gaussian_ll(mu_x, x.unsqueeze(dim=1).expand((B, K,) + x.shape[1:]),
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
            vis_3d_latents.append(utils.numpify(z.reshape(B, K, -1)))

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
            'x_recon': vis_recons,
            'x_comps': vis_comps,
            'hiers': vis_hiers,
            '2d_latents': vis_2d_latents,
            '3d_latents': vis_3d_latents,
            'scene_indices': list(tar['scn_id'][0].item() for tar in targets),
            'obs_views': obs_view_idx,
            'query_views': qry_view_idx
        }
        return preds