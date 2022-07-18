import os

import cv2
import numpy as np
import torch
from PIL import Image
from natsort import natsorted

import util
from blender_scripts.blender_render import BlenderRender
from evaluation import BaseEvaluator
from sketch.model_mse import SketchGenerator
from util import has_arg
from util.facescape_bs import FaceScapeBlendshape
from util.memcache import Memcached


def mc_read_mask(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img.ndim > 2:
        img = img[:, :, 0]
    return img


def dpmap_normalize(x):
    return ((x + 1) * 0.5 * 65535).astype(np.uint16)


class ScanImageEditingEvaluator(BaseEvaluator):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        super(ScanImageEditingEvaluator,
              ScanImageEditingEvaluator).modify_commandline_options(
            parser, is_train)
        if not has_arg(parser, '--root_path'):
            parser.add_argument('--root_path',
                                default=os.path.expanduser('~/DS-CHIA/data/detail_edit/curated_scan_image/'))
        if not has_arg(parser, '--use_mask'):
            parser.add_argument('--use_mask', type=util.str2bool, const=True, default=True)
        if not has_arg(parser, '--test_exp'):
            parser.add_argument('--test_exp', type=util.str2bool, const=True, default=True)
        if not has_arg(parser, '--test_age'):
            parser.add_argument('--test_age', type=util.str2bool, const=True, default=False)
        if not has_arg(parser, "--concat_age"):
            parser.add_argument("--concat_age", type=util.str2bool, const=True, default=False)
        if not has_arg(parser, '--no_render'):
            parser.add_argument('--no_render', type=util.str2bool, const=True, default=False)
        if not has_arg(parser, '--blender_path'):
            parser.add_argument('--blender_path', default='blender')
        return parser

    def __init__(self, opt, target_phase):
        super().__init__(opt, target_phase)
        self.target_ages = [0, 0.5, 1]
        self.sketch_gen = SketchGenerator('checkpoints/model_mse.pth')
        self.blender_render = BlenderRender(blender_path=self.opt.blender_path)

    def test_exp(self, model, save_path, seg_dpmap):
        os.makedirs(f'{save_path}/exp_edit', exist_ok=True)
        try:
            label_origin = model(seg_dpmap, command='compute_aux')  # (1, 30)
        except Exception as e:
            label_origin = FaceScapeBlendshape.get_bs_concat(['1_neutral']).clone()  # (1, 29)
        # no reason to compute only interested expressions. Compute all expressions
        target_exp = FaceScapeBlendshape.get_bs_concat(FaceScapeBlendshape.exp_list)  # (20, 29)
        label_edit = label_origin.expand((target_exp.shape[0], -1)).clone()  # (20, 30)
        label_edit[:, :FaceScapeBlendshape.get_bs_cnt()] = target_exp
        sp, gl = model(seg_dpmap, command='encode')
        sp = sp.expand(label_edit.shape[0], -1, -1, -1)
        gl = gl.expand(label_edit.shape[0], -1)
        sp_edit, gl_edit = model(sp, gl, label_edit, command='transform_exp')
        edit = model(sp_edit, gl_edit, command='decode')  # (20, 2, 256, 256)
        edit_np = edit.detach().cpu().numpy()
        for i, exp_name in enumerate(FaceScapeBlendshape.exp_list):
            cv2.imwrite(f'{save_path}/exp_edit/{exp_name}.png', dpmap_normalize(edit_np[i, -1]))
            if not self.opt.no_render:
                self.blender_render.render_exp_edit(save_path, exp_name)

    def test_age(self, model, save_path, seg_dpmap):
        os.makedirs(f'{save_path}/age_edit', exist_ok=True)
        try:
            label_origin = model(seg_dpmap, command='compute_aux')  # (1, 30)
        except Exception as e:
            label_origin = [FaceScapeBlendshape.get_bs_concat(['1_neutral']).clone()]  # (1, 29)
            if self.opt.concat_age:
                label_origin.append(label_origin[0].new_zeros((label_origin[0].shape[0], 1)))
            label_origin = torch.cat(label_origin, dim=1)
        label_edit = label_origin.expand((len(self.target_ages), -1)).clone()  # (20, 30)
        for i in range(len(self.target_ages)):
            label_edit[i, FaceScapeBlendshape.get_bs_cnt():] = self.target_ages[i]
        sp, gl = model(seg_dpmap, command='encode')
        sp = sp.expand(label_edit.shape[0], -1, -1, -1)
        gl = gl.expand(label_edit.shape[0], -1)
        try:
            sp_edit, gl_edit = model(sp, gl, label_edit, command='transform_age')  # use transform age if possible
        except Exception as e:
            sp_edit, gl_edit = model(sp, gl, label_edit, command='transform_exp')
        edit = model(sp_edit, gl_edit, command='decode')  # (20, 2, 256, 256)
        edit_np = edit.detach().cpu().numpy()
        for i, target_age in enumerate(self.target_ages):
            cv2.imwrite(f'{save_path}/age_edit/age_{target_age:.2f}.png', dpmap_normalize(edit_np[i, -1]))
            if not self.opt.no_render:
                self.blender_render.render_age_edit(save_path, target_age)

    def evaluate(self, model, dataset, nsteps):
        under_data = dataset.underlying_dataset
        noise_fixed = False
        savedir = self.opt.root_path
        for filename in natsorted(os.listdir(self.opt.root_path)):
            try:
                dpmap = Memcached.cv2_imread(f'{self.opt.root_path}/{filename}/dpmap.png')
                if self.opt.use_mask:
                    mask = mc_read_mask(f'./predef/front_mask_inverted.png')
                    mask = cv2.resize(mask, (dpmap.shape[0], dpmap.shape[1]))
                    mask = torch.from_numpy(mask).expand(dpmap.shape)
                    dpmap[mask == 0] = 32768
                seg = self.sketch_gen.dpmap2seg(dpmap)
                seg_bak = seg
                # dpmap = under_data.transform_A(Image.fromarray(dpmap))
                dpmap = under_data.transform_B(Image.fromarray(dpmap))
                seg = under_data.transform_A(Image.fromarray(seg))
                seg_dpmap = torch.cat([seg, dpmap], dim=0).unsqueeze(0)  # (1, 2, 256, 256)
                # seg_dpmap = dpmap.unsqueeze(0)
                if not noise_fixed:
                    model(seg_dpmap, command='fix_noise')
                    noise_fixed = True
                sp, gl = model(seg_dpmap, command='encode')
                recon = model(sp, gl, command='decode')
                os.makedirs(f'{savedir}/{filename}/recon', exist_ok=True)
                cv2.imwrite(f'{savedir}/{filename}/recon/seg.png', seg_bak)
                cv2.imwrite(f'{savedir}/{filename}/recon/recon.png',
                            dpmap_normalize(recon.detach().cpu().numpy()[0, -1]))
                if not self.opt.no_render:
                    self.blender_render.render_recon(f'{savedir}/{filename}')
                if self.opt.test_exp:
                    self.test_exp(model, f'{savedir}/{filename}', seg_dpmap)
                if self.opt.test_age:
                    self.test_age(model, f'{savedir}/{filename}', seg_dpmap)
            except Exception as e:
                print(filename, e)
                raise e
        return {}
