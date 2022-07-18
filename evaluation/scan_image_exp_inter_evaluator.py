import os

import cv2
import numpy as np
import torch
from PIL import Image

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


class ScanImageExpInterEvaluator(BaseEvaluator):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        super(ScanImageExpInterEvaluator,
              ScanImageExpInterEvaluator).modify_commandline_options(
            parser, is_train)
        if not has_arg(parser, '--file_paths'):
            parser.add_argument('--file_paths', required=True)
        if not has_arg(parser, '--target_exps'):
            parser.add_argument('--target_exps', required=True)
        if not has_arg(parser, '--use_mask'):
            parser.add_argument('--use_mask', type=util.str2bool, const=True, default=True)
        if not has_arg(parser, '--n_frame'):
            parser.add_argument('--n_frame', type=int, default=30)
        if not has_arg(parser, '--no_render'):
            parser.add_argument('--no_render', type=util.str2bool, const=True, default=False)
        if not has_arg(parser, '--blender_path'):
            parser.add_argument('--blender_path', default='blender')
        if not has_arg(parser, '--ffmpeg_path'):
            parser.add_argument('--ffmpeg_path', default='ffmpeg')
        return parser

    def __init__(self, opt, target_phase):
        super().__init__(opt, target_phase)
        self.sketch_gen = SketchGenerator('checkpoints/model_mse.pth')
        self.blender_render = BlenderRender(blender_path=self.opt.blender_path, ffmpeg_path=self.opt.ffmpeg_path)

    def evaluate(self, model, dataset, nsteps):
        under_data = dataset.underlying_dataset
        noise_fixed = False
        for file_path, target_exp_name in zip(self.opt.file_paths.split(','), self.opt.target_exps.split(',')):
            dpmap = Memcached.cv2_imread(f'{file_path}/dpmap.png')

            if self.opt.use_mask:
                mask = mc_read_mask(f'./predef/front_mask_inverted.png')
                mask = cv2.resize(mask, (dpmap.shape[0], dpmap.shape[1]))
                mask = torch.from_numpy(mask).expand(dpmap.shape)
                dpmap[mask == 0] = 32768

            seg = self.sketch_gen.dpmap2seg(dpmap)
            dpmap = under_data.transform_B(Image.fromarray(dpmap))
            seg = under_data.transform_A(Image.fromarray(seg))
            seg_dpmap = torch.cat([seg, dpmap], dim=0).unsqueeze(0)  # (1, 2, 256, 256)

            if not noise_fixed:
                model(seg_dpmap, command='fix_noise')
                noise_fixed = True
            sp, gl = model(seg_dpmap, command='encode')
            sp_edit, gl_edit = model(sp, gl, FaceScapeBlendshape.get_bs_concat([target_exp_name]),
                                     command='transform_exp')
            os.makedirs(f'{file_path}/exp_inter_{target_exp_name}', exist_ok=True)
            for i in range(self.opt.n_frame):
                ratio = i / (self.opt.n_frame - 1)
                sp_mid = torch.lerp(sp, sp_edit, ratio)
                gl_mid = torch.lerp(gl, gl_edit, ratio)
                edit_mid = model(sp_mid, gl_mid, command='decode')
                edit_mid_np = edit_mid.detach().cpu().numpy()
                cv2.imwrite(f'{file_path}/exp_inter_{target_exp_name}/frame_{i}.png',
                            dpmap_normalize(edit_mid_np[0, -1]))
            if not self.opt.no_render:
                self.blender_render.render_exp_inter(file_path, target_exp_name, self.opt.n_frame)
        return {}
