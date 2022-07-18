import lpips as lpips
import torch

import models.networks as networks
import models.networks.loss as loss
from evaluation.scan_image_editing_evaluator import mc_read_mask
from models.networks.stylegan2_layers import FeatureMatchingLoss
from models.swapping_autoencoder_model import SwappingAutoencoderModel
from util import util, has_arg


class SwappingAutoencoderDeltaModel(SwappingAutoencoderModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        super(SwappingAutoencoderDeltaModel, SwappingAutoencoderDeltaModel).modify_commandline_options(
            parser, is_train)
        parser.add_argument('--swap_l1_channel_index', type=int, default=0)
        parser.add_argument('--lambda_L1_mix', default=1.0, type=float)
        parser.add_argument('--lambda_L1_dpmap', default=1.0, type=float)
        parser.add_argument('--lambda_L1_masked', default=0.0, type=float)
        parser.add_argument('--lambda_L1_mix_masked', default=0.0, type=float)
        parser.add_argument('--lambda_L1_mix_dpmap', default=0.0, type=float)
        parser.add_argument('--roi_mask_path', default='./predef/mask_v13_256.png')
        parser.add_argument('--lambda_LPIPS_mix', default=1.0, type=float)
        parser.add_argument('--lambda_exp_sparse', default=0.0, type=float)
        parser.add_argument('--lambda_FM', default=0.0, type=float)
        if not has_arg(parser, '--cond_recon'):
            parser.add_argument("--cond_recon", type=util.str2bool, const=True, default=False)
        parser.add_argument('--replace_seg', type=util.str2bool, const=True, default=False)
        parser.add_argument('--replace_seg_ch', type=int, default=0)  # seg
        return parser

    def initialize(self):
        self.E = networks.create_network(self.opt, self.opt.netE, "encoder")
        self.G = networks.create_network(self.opt, self.opt.netG, "generator")
        self.TExp = networks.create_network(self.opt, self.opt.netTExp, 'transformer')
        if self.opt.lambda_GAN > 0.0:
            self.D = networks.create_network(
                self.opt, self.opt.netD, "discriminator")
        if self.opt.lambda_PatchGAN > 0.0:
            self.Dpatch = networks.create_network(
                self.opt, self.opt.netPatchD, "patch_discriminator"
            )

        # Count the iteration count of the discriminator
        # Used for lazy R1 regularization (c.f. Appendix B of StyleGAN2)
        self.register_buffer(
            "num_discriminator_iters", torch.zeros(1, dtype=torch.long)
        )
        self.register_buffer('roi_mask',
                             torch.from_numpy(mc_read_mask(self.opt.roi_mask_path)).float()[None, None, ...] / 255.0)
        self.l1_loss = torch.nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
        self.fm_loss = FeatureMatchingLoss()

        if (not self.opt.isTrain) or self.opt.continue_train:
            self.load()

        if self.opt.num_gpus > 0:
            self.to("cuda:0")

    def compute_R1_loss(self, images):
        real, real_swap, label = images
        return super().compute_R1_loss(real)

    def get_lpips_feature(self, x):
        return x[:, self.opt.lpips_channel_index][:, None, ...].repeat((1, 3, 1, 1))

    def transform_exp(self, sp, gl, exp_label):
        sp_trans, gl_trans = self.TExp(sp, gl, exp_label)
        return sp_trans, gl_trans

    def compute_generator_losses(self, images, sp_ma, gl_ma):
        real, real_swap, label = images
        losses, metrics = {}, {}
        B = real.size(0)

        sp, gl = self.E(real)
        if self.opt.cond_recon:
            sp_rec, gl_rec = self.transform_exp(sp[:B // 2], gl[:B // 2], label[:B // 2])
            rec = self.G(sp_rec, gl_rec)
        else:
            rec = self.G(sp[:B // 2], gl[:B // 2])  # only on B//2 to save memory
        sp_mix = self.swap(sp)
        gl_mix = self.swap(gl)
        sp_trans, gl_trans = self.transform_exp(sp_mix, gl_mix, label)

        mix = self.G(sp_trans, gl_trans)

        # record the error of the reconstructed images for monitoring purposes
        metrics["L1_dist"] = self.l1_loss(rec[:, self.opt.swap_l1_channel_index],
                                          real[:B // 2, self.opt.swap_l1_channel_index])
        metrics['L1_dist_dpmap'] = self.l1_loss(rec[:, self.opt.lpips_channel_index],
                                                real[:B // 2, self.opt.lpips_channel_index])
        metrics["LPIPS_dist"] = self.lpips_loss(self.get_lpips_feature(rec), self.get_lpips_feature(real[:B // 2]))
        metrics['L1_dist_mix'] = self.l1_loss(mix[:, self.opt.swap_l1_channel_index],
                                              real_swap[:, self.opt.swap_l1_channel_index])
        metrics['L1_dist_mix_dpmap'] = self.l1_loss(mix[:, self.opt.lpips_channel_index],
                                                    real_swap[:, self.opt.lpips_channel_index])
        metrics['LPIPS_dist_mix'] = self.lpips_loss(self.get_lpips_feature(mix), self.get_lpips_feature(real_swap))
        metrics['exp_dist_sparse'] = self.l1_loss(sp_trans, sp_mix)
        metrics["L1_dist_masked"] = self.l1_loss(rec[:, self.opt.swap_l1_channel_index] * self.roi_mask,
                                                 real[:B // 2, self.opt.swap_l1_channel_index] * self.roi_mask)
        metrics["L1_dist_mix_masked"] = self.l1_loss(mix[:, self.opt.swap_l1_channel_index] * self.roi_mask,
                                                     real_swap[:, self.opt.swap_l1_channel_index] * self.roi_mask)

        if self.opt.lambda_L1 > 0.0:
            losses["G_L1"] = metrics["L1_dist"] * self.opt.lambda_L1

        if self.opt.lambda_L1_masked > 0.0:
            losses["G_L1_masked"] = metrics["L1_dist_masked"] * self.opt.lambda_L1_masked

        if self.opt.lambda_L1_dpmap > 0.0:
            losses['G_L1_dpmap'] = metrics['L1_dist_dpmap'] * self.opt.lambda_L1_dpmap

        if self.opt.lambda_LPIPS > 0.0:
            losses['G_LPIPS'] = metrics['LPIPS_dist'] * self.opt.lambda_LPIPS

        if self.opt.lambda_L1_mix > 0.0:
            losses['G_L1_mix'] = metrics['L1_dist_mix'] * self.opt.lambda_L1_mix

        if self.opt.lambda_L1_mix_masked > 0.0:
            losses['G_L1_mix_masked'] = metrics['L1_dist_mix_masked'] * self.opt.lambda_L1_mix_masked

        if self.opt.lambda_L1_mix_dpmap > 0.0:
            losses['G_L1_mix_dpmap'] = metrics['L1_dist_mix_dpmap'] * self.opt.lambda_L1_mix_dpmap

        if self.opt.lambda_LPIPS_mix > 0.0:
            losses['G_LPIPS_mix'] = metrics['LPIPS_dist_mix'] * self.opt.lambda_LPIPS_mix

        if self.opt.lambda_exp_sparse > 0.0:
            losses['G_exp_sparse'] = metrics['exp_dist_sparse'] * self.opt.lambda_exp_sparse

        if self.opt.lambda_GAN > 0.0:
            if self.opt.lambda_FM > 0.0:
                pred_rec, feat_rec = self.D(self.replace_seg(rec, real[:B // 2]), extract_features=True)
                pred_mix, feat_mix = self.D(mix, extract_features=True)
            else:
                pred_rec = self.D(rec)
                pred_mix = self.D(mix)

            losses["G_GAN_rec"] = loss.gan_loss(
                pred_rec,
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 0.5)

            losses["G_GAN_mix"] = loss.gan_loss(
                pred_mix,
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 1.0)

            if self.opt.lambda_FM > 0.0:
                with torch.no_grad():
                    pred_real, feat_real = self.D(real[:B // 2], extract_features=True)
                    pred_real_swap, feat_real_swap = self.D(real_swap, extract_features=True)

                metrics['FM_dist'] = self.fm_loss(feat_rec, feat_real)
                losses['G_FM'] = metrics['FM_dist'] * self.opt.lambda_FM
                metrics['FM_dist_mix'] = self.fm_loss(feat_mix, feat_real_swap)
                losses['G_FM_mix'] = metrics['FM_dist_mix'] * self.opt.lambda_FM

        if self.opt.lambda_PatchGAN > 0.0:
            real_feat = self.Dpatch.extract_features(
                self.get_random_crops(real),
                aggregate=self.opt.patch_use_aggregation).detach()
            mix_feat = self.Dpatch.extract_features(self.get_random_crops(mix))

            losses["G_mix"] = loss.gan_loss(
                self.Dpatch.discriminate_features(real_feat, mix_feat),
                should_be_classified_as_real=True,
            ) * self.opt.lambda_PatchGAN

        return losses, metrics

    def replace_seg(self, rec, real):
        if self.opt.replace_seg:
            res = []
            for i in range(rec.shape[1]):
                if i == self.opt.replace_seg_ch:
                    res.append(real[:, i:i + 1])
                else:
                    res.append(rec[:, i:i + 1])
            res = torch.cat(res, dim=1)
            return res
        else:
            return rec

    def compute_image_discriminator_losses(self, real, rec, mix):
        if self.opt.lambda_GAN == 0.0:
            return {}

        B = real.size(0)
        pred_real = self.D(real)
        pred_rec = self.D(self.replace_seg(rec, real[:B // 2]))
        pred_mix = self.D(mix)

        losses = {}
        losses["D_real"] = loss.gan_loss(
            pred_real, should_be_classified_as_real=True
        ) * self.opt.lambda_GAN

        losses["D_rec"] = loss.gan_loss(
            pred_rec, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)
        losses["D_mix"] = loss.gan_loss(
            pred_mix, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)

        return losses

    def compute_discriminator_losses(self, images):
        self.num_discriminator_iters.add_(1)
        real, real_swap, label = images

        sp, gl = self.E(real)
        B = real.size(0)
        assert B % 2 == 0, "Batch size must be even on each GPU."

        # To save memory, compute the GAN loss on only
        # half of the reconstructed images
        if self.opt.cond_recon:
            sp_rec, gl_rec = self.transform_exp(sp[:B // 2], gl[:B // 2], label[:B // 2])
            rec = self.G(sp_rec, gl_rec)
        else:
            rec = self.G(sp[:B // 2], gl[:B // 2])  # only on B//2 to save memory
        sp_mix = self.swap(sp)
        gl_mix = self.swap(gl)
        sp_trans, gl_trans = self.transform_exp(sp_mix, gl_mix, label)
        mix = self.G(sp_trans, gl_trans)

        losses = self.compute_image_discriminator_losses(real, rec, mix)

        if self.opt.lambda_PatchGAN > 0.0:
            patch_losses = self.compute_patch_discriminator_losses(real, mix)
            losses.update(patch_losses)

        metrics = {}  # no metrics to report for the Discriminator iteration

        return losses, metrics, sp.detach(), gl.detach()

    def get_parameters_for_mode(self, mode):
        if mode == "generator":
            Gparams = list(self.G.parameters()) + list(self.E.parameters()) + list(self.TExp.parameters())
            return Gparams
        elif mode == "discriminator":
            Dparams = []
            if self.opt.lambda_GAN > 0.0:
                Dparams += list(self.D.parameters())
            if self.opt.lambda_PatchGAN > 0.0:
                Dparams += list(self.Dpatch.parameters())
            return Dparams

    def get_visuals_for_snapshot(self, images):
        real, real_swap, label = images

        if self.opt.isTrain:
            # avoid the overhead of generating too many visuals during training
            real = real[:2] if self.opt.num_gpus > 1 else real[:4]
            real_swap = real_swap[:2] if self.opt.num_gpus > 1 else real_swap[:4]
            label = label[:2] if self.opt.num_gpus > 1 else label[:4]
        sp, gl = self.E(real)
        sp_mix = self.swap(sp)
        gl_mix = self.swap(gl)
        sp_trans, gl_trans = self.transform_exp(sp_mix, gl_mix, label)
        if self.opt.cond_recon:
            sp_rec, gl_rec = self.transform_exp(sp, gl, label)
            rec = self.G(sp_rec, gl_rec)
        else:
            rec = self.G(sp, gl)
        mix = self.G(sp_trans, gl_trans)
        visuals = {"real": real, "real_swap": real_swap, "rec": rec, "mix": mix}
        if visuals['real'].shape[1] == 2:
            visuals['real'] = util.split_tensor(visuals['real'])
            visuals['rec'] = util.split_tensor(visuals['rec'])
            visuals['mix'] = util.split_tensor(visuals['mix'])
            visuals['real_swap'] = util.split_tensor(visuals['real_swap'])
        return visuals
