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
from util import has_arg, extract_id_name_exp_name
from util.facescape_bs import FaceScapeBlendshape
from util.memcache import Memcached


def mc_read_mask(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img.ndim > 2:
        img = img[:, :, 0]
    return img


def dpmap_normalize(x):
    return ((x + 1) * 0.5 * 65535).astype(np.uint16)


class ScanImageAgeProgressionEvaluator(BaseEvaluator):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        super(ScanImageAgeProgressionEvaluator,
              ScanImageAgeProgressionEvaluator).modify_commandline_options(
            parser, is_train)
        if not has_arg(parser, '--file_paths'):
            parser.add_argument('--file_paths', required=True)
        if not has_arg(parser, '--use_mask'):
            parser.add_argument('--use_mask', type=util.str2bool, const=True, default=True)
        if not has_arg(parser, '--n_frame'):
            parser.add_argument('--n_frame', type=int, default=180)
        if not has_arg(parser, '--anchor_num'):
            parser.add_argument('--anchor_num', type=int, default=-1, help='-1 means use specified age at every frame')
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
        for file_path in self.opt.file_paths.split(','):
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
            os.makedirs(f'{file_path}/age_progression', exist_ok=True)

            label = FaceScapeBlendshape.get_bs_concat(['1_neutral'])
            anchors = []
            anchor_ages = []

            if self.opt.anchor_num != -1:
                anchor_ages.append(0)
                if self.opt.anchor_num != 0:
                    anchor_ages.extend([(i + 0.5) / self.opt.anchor_num for i in range(self.opt.anchor_num)])
                anchor_ages.append(1)
            for anchor_age in anchor_ages:
                label[:, -1:] = anchor_age
                anchors.append(model(sp, gl, label, command='transform_age'))

            cur_anchor_idx = 0
            for i in range(self.opt.n_frame):
                target_age = i / (self.opt.n_frame - 1)
                if self.opt.anchor_num == -1:
                    label[:, -1:] = target_age
                    sp_age, gl_age = model(sp, gl, label, command='transform_age')
                else:
                    while cur_anchor_idx + 2 < len(anchor_ages) and target_age >= anchor_ages[cur_anchor_idx + 1]:
                        cur_anchor_idx += 1
                    inner_ratio = (target_age - anchor_ages[cur_anchor_idx]) / (
                            anchor_ages[cur_anchor_idx + 1] - anchor_ages[cur_anchor_idx])
                    sp_age = torch.lerp(anchors[cur_anchor_idx][0], anchors[cur_anchor_idx + 1][0], inner_ratio)
                    gl_age = torch.lerp(anchors[cur_anchor_idx][1], anchors[cur_anchor_idx + 1][1], inner_ratio)
                edit_age = model(sp_age, gl_age, command='decode')
                edit_age_np = edit_age.detach().cpu().numpy()
                cv2.imwrite(f'{file_path}/age_progression/frame_{i}.png', dpmap_normalize(edit_age_np[0, -1]))
            if not self.opt.no_render:
                self.blender_render.render_age_progression(file_path, self.opt.n_frame)
        return {}
