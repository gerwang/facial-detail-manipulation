import os

import cv2
import torch
from PIL import Image

import util
from blender_scripts.blender_render import BlenderRender
from evaluation import BaseEvaluator
from evaluation.scan_image_editing_evaluator import mc_read_mask, dpmap_normalize
from sketch.model_mse import SketchGenerator
from util import has_arg
from util.bs_coeff_reader import BlendshapeReader, detail_bs_indices
from util.facescape_bs import FaceScapeBlendshape
from util.memcache import Memcached


class ScanImageBsAnimeEvaluator(BaseEvaluator):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        super(ScanImageBsAnimeEvaluator,
              ScanImageBsAnimeEvaluator).modify_commandline_options(
            parser, is_train)
        if not has_arg(parser, '--use_mask'):
            parser.add_argument('--use_mask', type=util.str2bool, const=True, default=True)
        if not has_arg(parser, '--file_paths'):
            parser.add_argument('--file_paths', required=True)
        if not has_arg(parser, '--bs_clip_path'):
            parser.add_argument('--bs_clip_path', default=os.path.expanduser(
                './predef/2.txt'))
        if not has_arg(parser, '--use_fs_pro'):
            parser.add_argument('--use_fs_pro', type=util.str2bool, const=True, default=True)
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

    def get_bs_latent_code(self, sp, gl, model, label_origin, target_exp):
        label_edit = label_origin.clone()
        label_edit[:, :FaceScapeBlendshape.get_bs_cnt()] = target_exp
        return model(sp, gl, label_edit, command='transform_exp')

    def test_bs_anime(self, model, save_path, seg_dpmap, bs_reader):
        sp, gl = model(seg_dpmap, command='encode')
        try:
            label_origin = model(seg_dpmap, command='compute_aux')  # (1, 30)
        except Exception as e:
            label_origin = FaceScapeBlendshape.get_bs_concat(['1_neutral']).clone()  # (1, 29)
        os.makedirs(save_path, exist_ok=True)
        for j in range(len(bs_reader.bs)):
            np_exp = bs_reader.bs[j][detail_bs_indices][None, ...]
            target_exp = torch.from_numpy(np_exp).float()
            sp_frame, gl_frame = self.get_bs_latent_code(sp, gl, model, label_origin, target_exp)
            anime = model(sp_frame, gl_frame, command='decode')
            cv2.imwrite(f'{save_path}/{j}.png',
                        dpmap_normalize(anime.detach().cpu().numpy()[0, -1]))

    def evaluate(self, model, dataset, nsteps):
        under_data = dataset.underlying_dataset
        bs_reader = BlendshapeReader(self.opt.bs_clip_path, self.opt.use_fs_pro)
        clip_name = os.path.splitext(os.path.basename(self.opt.bs_clip_path))[0]
        bs_clip_start = None
        bs_clip_end = None
        if clip_name == '1':
            bs_clip_start = 0
            bs_clip_end = 420
        elif clip_name == '2':
            bs_clip_start = 0
            bs_clip_end = 270
        elif clip_name == '3':
            bs_clip_start = 0
            bs_clip_end = 240
        bs_reader.clip(bs_clip_start, bs_clip_end)
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
            self.test_bs_anime(model, f'{file_path}/bs_anime_{clip_name}', seg_dpmap, bs_reader)
            if not self.opt.no_render:
                self.blender_render.render_bs_anime(file_path, clip_name, self.opt.bs_clip_path, bs_clip_start,
                                                    bs_clip_end)
        return {}
