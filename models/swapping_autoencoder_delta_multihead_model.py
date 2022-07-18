import torch
import lpips

import models.networks as networks
import models.networks.loss as loss
from evaluation.scan_image_editing_evaluator import mc_read_mask
from models.swapping_autoencoder_delta_model import SwappingAutoencoderDeltaModel
from util import util
from util.facescape_bs import FaceScapeBlendshape
from models.networks.stylegan2_layers import FeatureMatchingLoss


class SwappingAutoencoderDeltaMultiheadModel(SwappingAutoencoderDeltaModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        super(SwappingAutoencoderDeltaMultiheadModel,
              SwappingAutoencoderDeltaMultiheadModel).modify_commandline_options(
            parser, is_train)
        parser.add_argument('--lambda_L1_cyc', type=float, default=1.0)
        parser.add_argument('--lambda_L1_cyc_masked', type=float, default=0.0)
        parser.add_argument('--lambda_L1_cyc_dpmap', type=float, default=1.0)
        parser.add_argument('--lambda_LPIPS_cyc', type=float, default=1.0)
        return parser

    def initialize(self):
        self.E = networks.create_network(self.opt, self.opt.netE, "encoder")
        self.G = networks.create_network(self.opt, self.opt.netG, "generator")
        self.TExp = networks.create_network(self.opt, self.opt.netTExp, 'transformer')
        self.TAge = networks.create_network(self.opt, self.opt.netTAge, 'transformer')
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

    def transform_age(self, sp, gl, age_label):
        sp_trans, gl_trans = self.TAge(sp, gl, age_label)
        return sp_trans, gl_trans

    def compute_R1_loss(self, images):
        real, real_swap, label = images
        losses = {}
        if self.opt.lambda_R1 > 0.0:
            real.requires_grad_()
            pred_real_exp, pred_real_age = self.D(real, label)
            pred_real_exp = pred_real_exp.sum()
            pred_real_age = pred_real_age.sum()
            grad_real, = torch.autograd.grad(
                outputs=(pred_real_exp + pred_real_age) * 0.5,
                inputs=[real],
                create_graph=True,
                retain_graph=True,
            )
            grad_real2 = grad_real.pow(2)
            dims = list(range(1, grad_real2.ndim))
            grad_penalty = grad_real2.sum(dims) * (self.opt.lambda_R1 * 0.5)
        else:
            grad_penalty = 0.0

        if self.opt.lambda_patch_R1 > 0.0:
            real_crop = self.get_random_crops(real).detach()
            real_crop.requires_grad_()
            target_crop = self.get_random_crops(real).detach()
            target_crop.requires_grad_()

            real_feat = self.Dpatch.extract_features(
                real_crop,
                aggregate=self.opt.patch_use_aggregation)
            target_feat = self.Dpatch.extract_features(target_crop)
            pred_real_patch = self.Dpatch.discriminate_features(
                real_feat, target_feat
            ).sum()

            grad_real, grad_target = torch.autograd.grad(
                outputs=pred_real_patch,
                inputs=[real_crop, target_crop],
                create_graph=True,
                retain_graph=True,
            )

            dims = list(range(1, grad_real.ndim))
            grad_crop_penalty = grad_real.pow(2).sum(dims) + \
                                grad_target.pow(2).sum(dims)
            grad_crop_penalty *= (0.5 * self.opt.lambda_patch_R1 * 0.5)
        else:
            grad_crop_penalty = 0.0

        losses["D_R1"] = grad_penalty + grad_crop_penalty

        return losses

    def compute_generator_losses(self, images, sp_ma, gl_ma):
        real, real_swap, label = images
        exp_label, age_label = torch.split(label, [FaceScapeBlendshape.get_bs_cnt(), 1], dim=1)
        losses, metrics = {}, {}
        B = real.size(0)

        sp, gl = self.E(real)
        rec = self.G(sp[:B // 2], gl[:B // 2])  # only on B//2 to save memory
        sp_mix = self.swap(sp)
        gl_mix = self.swap(gl)
        sp_trans, gl_trans = self.transform_exp(sp_mix, gl_mix, exp_label)
        mix = self.G(sp_trans, gl_trans)

        random_age_label = torch.rand(age_label.shape, device=age_label.device)
        sp_age, gl_age = self.transform_age(sp, gl, random_age_label)
        mix_age = self.G(sp_age, gl_age)

        sp_cyc, gl_cyc = self.E(mix_age)
        sp_cyc_trans, gl_cyc_trans = self.transform_age(sp_cyc, gl_cyc, age_label)
        cyc_mix = self.G(sp_cyc_trans, gl_cyc_trans)

        # record the error of the reconstructed images for monitoring purposes
        metrics["L1_dist"] = self.l1_loss(rec[:, self.opt.swap_l1_channel_index],
                                          real[:B // 2, self.opt.swap_l1_channel_index])
        metrics['L1_dist_dpmap'] = self.l1_loss(rec[:, self.opt.lpips_channel_index],
                                                real[:B // 2, self.opt.lpips_channel_index])
        metrics['L1_dist_mix'] = self.l1_loss(mix[:, self.opt.swap_l1_channel_index],
                                              real_swap[:, self.opt.swap_l1_channel_index])
        metrics['L1_dist_mix_dpmap'] = self.l1_loss(mix[:, self.opt.lpips_channel_index],
                                                    real_swap[:, self.opt.lpips_channel_index])
        metrics['sp_dist_sparse'] = self.l1_loss(sp_trans, sp_mix)
        metrics['gl_dist_sparse'] = self.l1_loss(gl_trans, gl_mix)

        metrics['L1_dist_cyc'] = self.l1_loss(cyc_mix[:, self.opt.swap_l1_channel_index],
                                              real[:, self.opt.swap_l1_channel_index])
        metrics['L1_dist_cyc_dpmap'] = self.l1_loss(cyc_mix[:, self.opt.lpips_channel_index],
                                                    real[:, self.opt.lpips_channel_index])
        metrics["LPIPS_dist"] = self.lpips_loss(self.get_lpips_feature(rec), self.get_lpips_feature(real[:B // 2]))

        if self.opt.lambda_L1 > 0.0:
            losses["G_L1"] = metrics["L1_dist"] * self.opt.lambda_L1

        if self.opt.lambda_L1_masked > 0.0:
            metrics["L1_dist_masked"] = self.l1_loss(rec[:, self.opt.swap_l1_channel_index] * self.roi_mask,
                                                     real[:B // 2, self.opt.swap_l1_channel_index] * self.roi_mask)
            losses["G_L1_masked"] = metrics["L1_dist_masked"] * self.opt.lambda_L1_masked

        if self.opt.lambda_L1_dpmap > 0.0:
            losses['G_L1_dpmap'] = metrics['L1_dist_dpmap'] * self.opt.lambda_L1_dpmap

        if self.opt.lambda_LPIPS > 0.0:
            losses['G_LPIPS'] = metrics['LPIPS_dist'] * self.opt.lambda_LPIPS

        if self.opt.lambda_L1_mix > 0.0:
            losses['G_L1_mix'] = metrics['L1_dist_mix'] * self.opt.lambda_L1_mix

        if self.opt.lambda_L1_mix_masked > 0.0:
            metrics["L1_dist_mix_masked"] = self.l1_loss(mix[:, self.opt.swap_l1_channel_index] * self.roi_mask,
                                                         real_swap[:, self.opt.swap_l1_channel_index] * self.roi_mask)
            losses['G_L1_mix_masked'] = metrics['L1_dist_mix_masked'] * self.opt.lambda_L1_mix_masked

        if self.opt.lambda_L1_mix_dpmap > 0.0:
            losses['G_L1_mix_dpmap'] = metrics['L1_dist_mix_dpmap'] * self.opt.lambda_L1_mix_dpmap

        if self.opt.lambda_LPIPS_mix > 0.0:
            metrics['LPIPS_dist_mix'] = self.lpips_loss(self.get_lpips_feature(mix), self.get_lpips_feature(real_swap))
            losses['G_LPIPS_mix'] = metrics['LPIPS_dist_mix'] * self.opt.lambda_LPIPS_mix

        if self.opt.lambda_exp_sparse > 0.0:
            losses['G_exp_sparse'] = metrics['sp_dist_sparse'] * self.opt.lambda_exp_sparse

        if self.opt.lambda_L1_cyc > 0.0:
            losses['G_L1_cyc'] = metrics['L1_dist_cyc'] * self.opt.lambda_L1_cyc

        if self.opt.lambda_L1_cyc_masked > 0.0:
            metrics['L1_dist_cyc_masked'] = self.l1_loss(cyc_mix[:, self.opt.swap_l1_channel_index] * self.roi_mask,
                                                         real[:, self.opt.swap_l1_channel_index] * self.roi_mask)
            losses['G_L1_cyc_masked'] = metrics['L1_dist_cyc_masked'] * self.opt.lambda_L1_cyc_masked

        if self.opt.lambda_L1_cyc_dpmap > 0.0:
            losses['G_L1_cyc_dpmap'] = metrics['L1_dist_cyc_dpmap'] * self.opt.lambda_L1_cyc_dpmap

        if self.opt.lambda_LPIPS_cyc > 0.0:
            metrics['LPIPS_dist_cyc'] = self.lpips_loss(self.get_lpips_feature(cyc_mix), self.get_lpips_feature(real))
            losses['G_LPIPS_cyc'] = metrics['LPIPS_dist_cyc'] * self.opt.lambda_LPIPS_cyc

        if self.opt.lambda_GAN > 0.0:
            pred_mix_age_exp, pred_mix_age_age = self.D(mix_age, torch.cat([exp_label, random_age_label], dim=1))
            if self.opt.lambda_FM > 0.0:
                pred_rec_exp, pred_rec_age, feat_rec = self.D(rec, label[:B // 2], extract_features=True)
                pred_mix_exp, pred_mix_age, feat_mix = self.D(mix, label, extract_features=True)
                _, _, feat_cyc_mix = self.D(cyc_mix[:B // 2], label[:B // 2], extract_features=True)
            else:
                pred_rec_exp, pred_rec_age = self.D(rec, label[:B // 2])
                pred_mix_exp, pred_mix_age = self.D(mix, label)

            losses["G_GAN_rec"] = (loss.gan_loss(
                pred_rec_exp,
                should_be_classified_as_real=True
            ) + loss.gan_loss(
                pred_rec_age,
                should_be_classified_as_real=True
            )) * 0.5 * (self.opt.lambda_GAN * 0.5)

            losses["G_GAN_mix"] = (loss.gan_loss(
                pred_mix_exp,
                should_be_classified_as_real=True
            ) + loss.gan_loss(
                pred_mix_age,
                should_be_classified_as_real=True
            )) * 0.5 * (self.opt.lambda_GAN * 1.0)

            losses["G_GAN_mix_age"] = (loss.gan_loss(
                pred_mix_age_exp,
                should_be_classified_as_real=True
            ) + loss.gan_loss(
                pred_mix_age_age,
                should_be_classified_as_real=True
            )) * 0.5 * (self.opt.lambda_GAN * 1.0)

            if self.opt.lambda_FM > 0.0:
                with torch.no_grad():
                    pred_real_exp, pred_real_age, feat_real = self.D(real[:B // 2], label[:B // 2],
                                                                     extract_features=True)
                    pred_real_swap_exp, pred_real_swap_age, feat_real_swap = self.D(real_swap, label,
                                                                                    extract_features=True)

                metrics['FM_dist'] = self.fm_loss(feat_rec, feat_real)
                losses['G_FM'] = metrics['FM_dist'] * self.opt.lambda_FM
                metrics['FM_dist_mix'] = self.fm_loss(feat_mix, feat_real_swap)
                losses['G_FM_mix'] = metrics['FM_dist_mix'] * self.opt.lambda_FM
                metrics['FM_dist_mix_age'] = self.fm_loss(feat_cyc_mix, feat_real)
                losses['G_FM_mix_age'] = metrics['FM_dist_mix_age'] * self.opt.lambda_FM

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

    def compute_image_discriminator_losses(self, real, rec, mix, mix_age, label, modified_age_label):
        if self.opt.lambda_GAN == 0.0:
            return {}

        B = real.size(0)
        pred_real_exp, pred_real_age = self.D(real, label)
        pred_rec_exp, pred_rec_age = self.D(rec, label[:B // 2])
        pred_mix_exp, pred_mix_age = self.D(mix, label)
        pred_mix_age_exp, pred_mix_age_age = self.D(mix_age, modified_age_label)

        losses = {}
        losses["D_real"] = (loss.gan_loss(
            pred_real_exp, should_be_classified_as_real=True
        ) + loss.gan_loss(
            pred_real_age, should_be_classified_as_real=True
        )) * 0.5 * (1.5 * self.opt.lambda_GAN)

        losses["D_rec"] = (loss.gan_loss(
            pred_rec_exp, should_be_classified_as_real=False
        ) + loss.gan_loss(
            pred_rec_age, should_be_classified_as_real=False
        )) * 0.5 * (0.5 * self.opt.lambda_GAN)
        losses["D_mix"] = (loss.gan_loss(
            pred_mix_exp, should_be_classified_as_real=False
        ) + loss.gan_loss(
            pred_mix_age, should_be_classified_as_real=False
        )) * 0.5 * (0.5 * self.opt.lambda_GAN)
        losses["D_mix_age"] = (loss.gan_loss(
            pred_mix_age_exp, should_be_classified_as_real=False
        ) + loss.gan_loss(
            pred_mix_age_age, should_be_classified_as_real=False
        )) * 0.5 * (0.5 * self.opt.lambda_GAN)

        return losses

    def compute_discriminator_losses(self, images):
        self.num_discriminator_iters.add_(1)
        real, real_swap, label = images
        exp_label, age_label = torch.split(label, [FaceScapeBlendshape.get_bs_cnt(), 1], dim=1)

        B = real.size(0)
        assert B % 2 == 0, "Batch size must be even on each GPU."

        # To save memory, compute the GAN loss on only
        # half of the reconstructed images
        sp, gl = self.E(real)
        rec = self.G(sp[:B // 2], gl[:B // 2])  # only on B//2 to save memory
        sp_mix = self.swap(sp)
        gl_mix = self.swap(gl)
        sp_trans, gl_trans = self.transform_exp(sp_mix, gl_mix, exp_label)
        mix = self.G(sp_trans, gl_trans)

        random_age_label = torch.rand(age_label.shape, device=age_label.device)
        sp_age, gl_age = self.transform_age(sp, gl, random_age_label)
        mix_age = self.G(sp_age, gl_age)

        losses = self.compute_image_discriminator_losses(real, rec, mix, mix_age, label,
                                                         torch.cat([exp_label, random_age_label], dim=1))

        if self.opt.lambda_PatchGAN > 0.0:
            patch_losses = self.compute_patch_discriminator_losses(real, mix)
            losses.update(patch_losses)

        metrics = {}  # no metrics to report for the Discriminator iteration

        return losses, metrics, sp.detach(), gl.detach()

    def get_visuals_for_snapshot(self, images):
        real, real_swap, label = images

        if self.opt.isTrain:
            # avoid the overhead of generating too many visuals during training
            real = real[:2] if self.opt.num_gpus > 1 else real[:4]
            real_swap = real_swap[:2] if self.opt.num_gpus > 1 else real_swap[:4]
            label = label[:2] if self.opt.num_gpus > 1 else label[:4]

        exp_label, age_label = torch.split(label, [FaceScapeBlendshape.get_bs_cnt(), 1], dim=1)

        sp, gl = self.E(real)
        rec = self.G(sp, gl)
        sp_mix = self.swap(sp)
        gl_mix = self.swap(gl)
        sp_trans, gl_trans = self.transform_exp(sp_mix, gl_mix, exp_label)
        mix = self.G(sp_trans, gl_trans)

        random_age_label = torch.rand(age_label.shape, device=age_label.device)
        sp_age, gl_age = self.transform_age(sp, gl, random_age_label)
        mix_age = self.G(sp_age, gl_age)

        sp_cyc, gl_cyc = self.E(mix_age)
        sp_cyc_trans, gl_cyc_trans = self.transform_age(sp_cyc, gl_cyc, age_label)
        cyc_mix = self.G(sp_cyc_trans, gl_cyc_trans)

        visuals = {"real": real, "real_swap": real_swap, "rec": rec, "mix": mix, 'mix_age': mix_age, 'cyc': cyc_mix}
        visuals['real'] = util.split_tensor(visuals['real'])
        visuals['rec'] = util.split_tensor(visuals['rec'])
        visuals['mix'] = util.split_tensor(visuals['mix'])
        visuals['real_swap'] = util.split_tensor(visuals['real_swap'])
        visuals['mix_age'] = util.split_tensor(visuals['mix_age'])
        visuals['cyc'] = util.split_tensor(visuals['cyc'])
        return visuals
