import numpy as np
import torch
from PIL import Image

import util
from evaluation.swap_visualization_delta_evaluator import SwapVisualizationDeltaEvaluator
from util.facescape_bs import FaceScapeBlendshape


class SwapVisualizationDeltaAgeOnlyEvaluator(SwapVisualizationDeltaEvaluator):
    def __init__(self, opt, target_phase):
        super().__init__(opt, target_phase)
        self.only_type = 'age'

    def generate_mix_grid(self, model, images):
        sps, gls, labels = [], [], []
        for image_label in images:
            image, label = image_label
            assert image.size(0) == 1
            sp, gl = model(image.expand(self.opt.num_gpus, -1, -1, -1), command="encode")
            sp = sp[:1]
            gl = gl[:1]
            sps.append(sp)
            gls.append(gl)
            labels.append(label)
        gl = torch.cat(gls, dim=0)
        sp = torch.cat(sps, dim=0)

        def put_img(img, canvas, row, col):
            h, w = img.shape[0], img.shape[1]
            start_x = w * col
            start_y = h * row
            canvas[start_y:start_y + h, start_x: start_x + w] = img

        images_np = util.tensor2im(images[0][0], tile=False)[0]
        grid_w = images_np.shape[1] * (gl.size(0) + 1)
        grid_h = images_np.shape[0] * (gl.size(0) + 1)
        if images_np.shape[2] == 2:  # both channel
            grid_img = np.ones((grid_h, grid_w * 2, 3), dtype=np.uint8)
        else:
            grid_img = np.ones((grid_h, grid_w, 3), dtype=np.uint8)
        for i, image_label in enumerate(images):
            image, label = image_label
            image_np = util.tensor2im(image, tile=False)[0]
            if image_np.shape[2] == 2:
                image_np = np.concatenate([image_np[:, :, 0:1], image_np[:, :, 1:2]], axis=1)
            put_img(image_np, grid_img, 0, i + 1)
            put_img(image_np, grid_img, i + 1, 0)

        for i, label in enumerate(labels):
            label_for_current_row = label.repeat((gl.size(0),) + tuple(1 for _ in range(label.ndimension() - 1)))
            if self.only_type == 'age':
                label_for_current_row[:, :FaceScapeBlendshape.get_bs_cnt()] = torch.cat(labels)[:,
                                                                              :FaceScapeBlendshape.get_bs_cnt()]
            elif self.only_type == 'exp':
                label_for_current_row[:, FaceScapeBlendshape.get_bs_cnt():] = torch.cat(labels)[:,
                                                                              FaceScapeBlendshape.get_bs_cnt():]
            else:
                raise ValueError(f'Unknown {self.only_type}')
            if self.only_type == 'age':
                try:
                    sp_for_current_row, gl_for_current_row = model(sp, gl, label_for_current_row,
                                                                   command='transform_age')
                except Exception as e:
                    sp_for_current_row, gl_for_current_row = model(sp, gl, label_for_current_row,
                                                                   command='transform_exp')
            else:
                sp_for_current_row, gl_for_current_row = model(sp, gl, label_for_current_row, command='transform_exp')
            mix_row = model(sp_for_current_row, gl_for_current_row, command="decode")
            mix_row = util.tensor2im(mix_row, tile=False)
            for j, mix in enumerate(mix_row):
                if mix.shape[2] == 2:
                    mix = np.concatenate([mix[:, :, 0:1], mix[:, :, 1:2]], axis=1)
                put_img(mix, grid_img, i + 1, j + 1)

        final_grid = Image.fromarray(grid_img)
        return final_grid
