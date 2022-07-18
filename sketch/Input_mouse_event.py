# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
import torch
from PIL import Image
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import models
from data.seg_dpmap_dataset import get_dt_transforms, get_dpmap_transforms
from datetime import datetime

from evaluation.scan_image_editing_evaluator import dpmap_normalize


class InputGraphicsScene(QGraphicsScene):
    def __init__(self, mode_list, paint_size, up_sketch_view, sketch_gen, opt, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.modes = mode_list
        self.mouse_clicked = False
        self.prev_pt = None
        self.setSceneRect(0, 0, self.width(), self.height())

        self.sketch_gen = sketch_gen

        # save the points
        self.mask_points = []
        self.sketch_points = []
        self.stroke_points = []
        self.image_list = []

        self.up_sketch_view = up_sketch_view

        # save the history of edit
        self.history = []
        self.sample_Num = 15
        self.refine = True

        self.sketch_img = np.ones((512, 512, 3), dtype=np.float32)
        self.ori_img = np.ones((512, 512, 3), dtype=np.uint8) * 255
        self.image_list.append(self.sketch_img.copy())
        self.generated = np.ones((512, 512, 3), dtype=np.uint8) * 255
        # strokes color
        self.stk_color = None
        self.paint_size = paint_size
        self.paint_color = (0, 0, 0)
        # self.setPos(0 ,0)

        self.convert_on = False
        self.mouse_up = False

        self.black_value = 0.0

        self.iter = 0
        self.max_iter = 20
        self.firstDisplay = True
        self.mouse_released = False

        self.updatePixmap(True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updatePixmap)
        self.timer.start(10)
        self.opt = opt
        self.model = models.create_model(opt)

        old_input_nc = opt.input_nc
        opt.input_nc //= 2
        self.dpmap_transforms = get_dpmap_transforms(opt)
        self.dt_transforms = get_dt_transforms(opt)
        opt.input_nc = old_input_nc

        self.orig_dpmap = None  # dpmap data, in torch.Tensor
        self.gen_dpmap = None
        self.noise_fixed = False
        self.dpmap_modes = ['always_original', 'change_each_time']
        self.current_dpmap_mode = 'always_original'

        self.result_path = None

    def vis_dpmap(self, img):
        """
        Args:
            img: (256, 256) float dpmap

        Returns: (512, 512, 3) RGB image

        """
        img = ((img + 1) * 0.5 * 255).astype(np.uint8)
        img = np.repeat(img[..., None], 3, 2)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        return img

    def seg2sketch(self, img):
        img = np.repeat((1 - img.astype(np.float32) / 255)[..., None], 3, 2)  # invert black and white
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        return img

    def sketch2seg(self, img):
        img = ((1 - img[:, :, 0]) * 255).astype(np.uint8)
        img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size), interpolation=cv2.INTER_NEAREST)
        return img

    def generate_dpmap(self, save_file=True):
        """

        Returns: self.gen_dpmap original torch tensor
        self.generated RGB [512, 512, 3] image

        """
        if self.orig_dpmap is None:
            QMessageBox.information(self.parent(), "Image Draw", "Haven't loaded dpmap yet")
            return
        with torch.no_grad():
            seg_dpmap = torch.cat(
                [self.dt_transforms(Image.fromarray(self.sketch2seg(self.sketch_img))), self.orig_dpmap],
                dim=0).unsqueeze(0)
            if not self.noise_fixed:
                self.model(seg_dpmap, command='fix_noise')
                self.noise_fixed = True
            sp, gl = self.model(seg_dpmap, command='encode')
            self.gen_dpmap = self.model(sp, gl, command='decode')[0, 1].cpu().numpy()
            if self.current_dpmap_mode == 'change_each_time':
                self.orig_dpmap = torch.from_numpy(self.gen_dpmap).unsqueeze(0)
            self.generated = self.vis_dpmap(self.gen_dpmap)

        if save_file:
            self.saveCurrentImage()

    def reset(self):
        # save the points
        self.mask_points = []
        self.sketch_points = []
        self.stroke_points = []

        self.sketch_img = np.ones((512, 512, 3), dtype=np.float32)
        self.ori_img = np.ones((512, 512, 3), dtype=np.uint8) * 255
        self.generated = np.ones((512, 512, 3), dtype=np.uint8) * 255

        # save the history of edit
        self.history = []
        self.image_list.clear()
        self.image_list.append(self.sketch_img.copy())

        self.updatePixmap(True)
        self.convert_RGB()

        self.prev_pt = None
        self.orig_dpmap = None
        self.result_path = None

    def setSketchImag(self, dpmap_img, seg_img, fileName):
        self.reset()
        self.orig_dpmap = self.dpmap_transforms(Image.fromarray(dpmap_img))
        self.sketch_img = self.seg2sketch(seg_img)

        self.result_path = f'./saveImage/{datetime.now()}'
        os.makedirs(self.result_path, exist_ok=True)
        with open(f'{self.result_path}/fileName.txt', 'w') as f:
            f.write(fileName)

        self.generate_dpmap(save_file=False)
        self.sketch_img = self.seg2sketch(self.sketch_gen.dpmap2seg(dpmap_normalize(self.gen_dpmap)))
        self.generate_dpmap()
        # self.sketch_img = sketch_mat
        self.updatePixmap()
        self.image_list.clear()
        self.image_list.append(self.sketch_img.copy())

    def saveCurrentImage(self):
        cur_time = str(datetime.now())
        cv2.imwrite(f'{self.result_path}/seg_{cur_time}.png',
                    ((1 - self.sketch_img) * 255)[:, :, 0].astype(np.uint8))
        cv2.imwrite(f'{self.result_path}/dpmap_{cur_time}.png',
                    dpmap_normalize(self.gen_dpmap))

    def mousePressEvent(self, event):
        self.mouse_clicked = True
        self.prev_pt = None
        self.draw = False

    def mouseReleaseEvent(self, event):
        # print('Leave')
        if self.draw:
            self.image_list.append(self.sketch_img.copy())
            self.generate_dpmap()
            self.updatePixmap(True)

        self.draw = False
        self.prev_pt = None
        self.mouse_clicked = False
        self.mouse_released = True
        self.mouse_up = True

    def mouseMoveEvent(self, event):
        if self.mouse_clicked:
            if int(event.scenePos().x()) < 0 or int(event.scenePos().x()) > 512 or int(event.scenePos().y()) < 0 or int(
                    event.scenePos().y()) > 512:
                return
            if self.prev_pt and int(event.scenePos().x()) == self.prev_pt.x() and int(
                    event.scenePos().y()) == self.prev_pt.y():
                return
            if self.prev_pt:
                # self.drawSketch(self.prev_pt, event.scenePos())
                pts = {}
                pts['prev'] = (int(self.prev_pt.x()), int(self.prev_pt.y()))
                pts['curr'] = (int(event.scenePos().x()), int(event.scenePos().y()))
                # self.sketch_points.append(pts)
                self.make_sketch([pts])
                # self.history.append(1)
                self.prev_pt = event.scenePos()
            else:
                self.prev_pt = event.scenePos()

    def make_sketch(self, pts):
        if len(pts) > 0:
            for pt in pts:
                cv2.line(self.sketch_img, pt['prev'], pt['curr'], self.paint_color, self.paint_size)
        self.updatePixmap()
        self.draw = True
        self.iter = self.iter + 1
        if self.iter > self.max_iter:
            self.iter = 0

    def get_stk_color(self, color):
        self.stk_color = color

    def erase_prev_pt(self):
        self.prev_pt = None

    def reset_items(self):
        for i in range(len(self.items())):
            item = self.items()[0]
            self.removeItem(item)

    def undo(self):
        if len(self.image_list) > 1:
            num = len(self.image_list) - 2
            self.sketch_img = self.image_list[num].copy()
            self.image_list.pop(num + 1)

        self.generate_dpmap()
        self.updatePixmap(True)

    def getImage(self):
        return (self.sketch_img * self.ori_img).astype(np.uint8)

    def updatePixmap(self, mouse_up=False):
        # print('update')
        self.mouse_released = False

        sketch = (self.sketch_img * 255).astype(np.uint8)
        qim = QImage(sketch.data, sketch.shape[1], sketch.shape[0], QImage.Format_RGB888)
        if self.firstDisplay:
            self.reset_items()
            self.imItem = self.addPixmap(QPixmap.fromImage(qim))
            self.firstDispla = False
        else:
            self.imItem.setPixmap(QPixmap.fromImage(qim))

        if self.convert_on:
            self.convert_RGB()
            self.up_sketch_view.updatePixmap()

    def convert_RGB(self):
        self.up_sketch_view.setSketchImag(self.generated, True)
