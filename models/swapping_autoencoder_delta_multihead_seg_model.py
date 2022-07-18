import torch

import models.networks.loss as loss
from models.swapping_autoencoder_delta_multihead_model import SwappingAutoencoderDeltaMultiheadModel
from util import util
from util.facescape_bs import FaceScapeBlendshape


class SwappingAutoencoderDeltaMultiheadSegModel(SwappingAutoencoderDeltaMultiheadModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        super(SwappingAutoencoderDeltaMultiheadSegModel,
              SwappingAutoencoderDeltaMultiheadSegModel).modify_commandline_options(
            parser, is_train)
        parser.add_argument('--lambda_L1_seg', type=float, default=1.0)
        parser.add_argument('--mix_seg_label_mode', choices=['seg_all', 'seg_exp_dpmap_age'], default='seg_all')
        return parser

    def get_mix_seg_label(self, label):
        if self.opt.mix_seg_label_mode == 'seg_all':
            return label
        elif self.opt.mix_seg_label_mode == 'seg_exp_dpmap_age':
            exp_label, age_label = torch.split(label, [FaceScapeBlendshape.get_bs_cnt(), 1], dim=1)
            return torch.cat([exp_label, self.swap(age_label)])
        else:
            raise ValueError(f'unknown {self.opt.mix_seg_label_mode}')

    def compute_generator_losses(self, images, sp_ma, gl_ma):
        real, real_swap, label = images
        exp_label, age_label = torch.split(label, [FaceScapeBlendshape.get_bs_cnt(), 1], dim=1)
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

        random_age_label = torch.rand(age_label.shape, device=age_label.device)
        sp_age, gl_age = self.transform_age(sp, gl, torch.cat([exp_label, random_age_label], dim=1))
        mix_age = self.G(sp_age, gl_age)

        sp_cyc, gl_cyc = self.E(mix_age)
        sp_cyc_trans, gl_cyc_trans = self.transform_age(sp_cyc, gl_cyc, label)
        cyc_mix = self.G(sp_cyc_trans, gl_cyc_trans)

        seg_real, dpmap_real = torch.split(real, [1, 1], dim=1)
        dpmap_real_mix = self.swap(dpmap_real)
        real_mix = torch.cat([seg_real, dpmap_real_mix], dim=1)
        sp_seg, gl_seg = self.E(real_mix)
        if self.opt.cond_recon:
            sp_seg_cond, gl_seg_cond = self.transform_exp(sp_seg, gl_seg, self.get_mix_seg_label(label))
            mix_seg = self.G(sp_seg_cond, gl_seg_cond)
        else:
            mix_seg = self.G(sp_seg, gl_seg)

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
        # metrics['gl_dist_sparse'] = self.l1_loss(gl_trans, gl_mix)

        metrics['L1_dist_cyc'] = self.l1_loss(cyc_mix[:, self.opt.swap_l1_channel_index],
                                              real[:, self.opt.swap_l1_channel_index])
        metrics['L1_dist_cyc_dpmap'] = self.l1_loss(cyc_mix[:, self.opt.lpips_channel_index],
                                                    real[:, self.opt.lpips_channel_index])
        metrics["LPIPS_dist"] = self.lpips_loss(self.get_lpips_feature(rec), self.get_lpips_feature(real[:B // 2]))

        metrics["L1_dist_seg"] = self.l1_loss(mix_seg[:, self.opt.swap_l1_channel_index],
                                              real_mix[:, self.opt.swap_l1_channel_index])

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

        if self.opt.lambda_L1_seg > 0.0:
            losses['G_L1_seg'] = metrics['L1_dist_seg'] * self.opt.lambda_L1_seg

        if self.opt.lambda_GAN > 0.0:
            pred_mix_age_exp, pred_mix_age_age = self.D(mix_age, torch.cat([exp_label, random_age_label], dim=1))
            pred_mix_seg_exp, pred_mix_seg_age = self.D(mix_seg, self.get_mix_seg_label(label))
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

            losses["G_GAN_mix_seg"] = (loss.gan_loss(
                pred_mix_seg_exp,
                should_be_classified_as_real=True
            ) + loss.gan_loss(
                pred_mix_seg_age,
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

    def compute_image_discriminator_losses(self, real, rec, mix, mix_age, mix_seg, label, modified_age_label):
        if self.opt.lambda_GAN == 0.0:
            return {}

        B = real.size(0)
        pred_real_exp, pred_real_age = self.D(real, label)
        pred_rec_exp, pred_rec_age = self.D(rec, label[:B // 2])
        pred_mix_exp, pred_mix_age = self.D(mix, label)
        pred_mix_age_exp, pred_mix_age_age = self.D(mix_age, modified_age_label)
        pred_mix_seg_exp, pred_mix_seg_age = self.D(mix_seg, self.get_mix_seg_label(label))

        losses = {}
        losses["D_real"] = (loss.gan_loss(
            pred_real_exp, should_be_classified_as_real=True
        ) + loss.gan_loss(
            pred_real_age, should_be_classified_as_real=True
        )) * 0.5 * (2.0 * self.opt.lambda_GAN)

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
        losses["D_mix_seg"] = (loss.gan_loss(
            pred_mix_seg_exp, should_be_classified_as_real=False
        ) + loss.gan_loss(
            pred_mix_seg_age, should_be_classified_as_real=False
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
        if self.opt.cond_recon:
            sp_rec, gl_rec = self.transform_exp(sp[:B // 2], gl[:B // 2], label[:B // 2])
            rec = self.G(sp_rec, gl_rec)
        else:
            rec = self.G(sp[:B // 2], gl[:B // 2])  # only on B//2 to save memory
        sp_mix = self.swap(sp)
        gl_mix = self.swap(gl)
        sp_trans, gl_trans = self.transform_exp(sp_mix, gl_mix, label)
        mix = self.G(sp_trans, gl_trans)

        random_age_label = torch.rand(age_label.shape, device=age_label.device)
        sp_age, gl_age = self.transform_age(sp, gl, torch.cat([exp_label, random_age_label], dim=1))
        mix_age = self.G(sp_age, gl_age)

        seg_real, dpmap_real = torch.split(real, [1, 1], dim=1)
        dpmap_real_mix = self.swap(dpmap_real)
        real_mix = torch.cat([seg_real, dpmap_real_mix], dim=1)
        sp_seg, gl_seg = self.E(real_mix)
        if self.opt.cond_recon:
            sp_seg_cond, gl_seg_cond = self.transform_exp(sp_seg, gl_seg, self.get_mix_seg_label(label))
            mix_seg = self.G(sp_seg_cond, gl_seg_cond)
        else:
            mix_seg = self.G(sp_seg, gl_seg)

        losses = self.compute_image_discriminator_losses(real, rec, mix, mix_age, mix_seg, label,
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
        if self.opt.cond_recon:
            sp_rec, gl_rec = self.transform_exp(sp, gl, label)
            rec = self.G(sp_rec, gl_rec)
        else:
            rec = self.G(sp, gl)
        sp_mix = self.swap(sp)
        gl_mix = self.swap(gl)
        sp_trans, gl_trans = self.transform_exp(sp_mix, gl_mix, label)
        mix = self.G(sp_trans, gl_trans)

        random_age_label = torch.rand(age_label.shape, device=age_label.device)
        sp_age, gl_age = self.transform_age(sp, gl, torch.cat([exp_label, random_age_label], dim=1))
        mix_age = self.G(sp_age, gl_age)

        sp_cyc, gl_cyc = self.E(mix_age)
        sp_cyc_trans, gl_cyc_trans = self.transform_age(sp_cyc, gl_cyc, label)
        cyc_mix = self.G(sp_cyc_trans, gl_cyc_trans)

        seg_real, dpmap_real = torch.split(real, [1, 1], dim=1)
        dpmap_real_mix = self.swap(dpmap_real)
        real_mix = torch.cat([seg_real, dpmap_real_mix], dim=1)
        sp_seg, gl_seg = self.E(real_mix)
        if self.opt.cond_recon:
            sp_seg_cond, gl_seg_cond = self.transform_exp(sp_seg, gl_seg, self.get_mix_seg_label(label))
            mix_seg = self.G(sp_seg_cond, gl_seg_cond)
        else:
            mix_seg = self.G(sp_seg, gl_seg)

        visuals = {"real": real, "real_swap": real_swap, "rec": rec, "mix": mix, 'mix_age': mix_age, 'cyc': cyc_mix,
                   'mix_seg': mix_seg}
        visuals['real'] = util.split_tensor(visuals['real'])
        visuals['rec'] = util.split_tensor(visuals['rec'])
        visuals['mix'] = util.split_tensor(visuals['mix'])
        visuals['real_swap'] = util.split_tensor(visuals['real_swap'])
        visuals['mix_age'] = util.split_tensor(visuals['mix_age'])
        visuals['cyc'] = util.split_tensor(visuals['cyc'])
        visuals['mix_seg'] = util.split_tensor(visuals['mix_seg'])
        return visuals
