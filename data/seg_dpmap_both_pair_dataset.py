import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import util
from data.seg_dpmap_both_dataset import SegDpmapBothDataset
from data.seg_dpmap_dataset import SegDpmapDataset
from util import extract_id_name_exp_name
from util.facescape_bs import FaceScapeBlendshape
from util.memcache import Memcached


class SegDpmapBothPairDataset(SegDpmapBothDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = super(SegDpmapBothPairDataset, SegDpmapBothPairDataset).modify_commandline_options(parser, is_train)
        parser.add_argument('--replace_path')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        if opt.replace_path is not None:
            self.replace_img = cv2.imread(opt.replace_path, cv2.IMREAD_UNCHANGED)
            assert self.replace_img is not None
        else:
            self.replace_img = None

    def __getitem__(self, index):
        A_paths = self.A_paths_dict[self.current_phase]
        A_path = A_paths[index % len(A_paths)]
        B_paths = self.B_paths_dict[self.current_phase]
        A_seg_path = B_paths[index % len(B_paths)]
        A_id_name, A_exp_name = extract_id_name_exp_name(A_path)
        if self.current_phase == 'train':
            rand_index = np.random.randint(0, len(A_paths) - 1)
        else:
            rand_index = self.get_index_by_id_name_exp_name(A_id_name, '1_neutral')
        B_path = A_paths[rand_index]
        B_seg_path = B_paths[rand_index]
        B_id_name, B_exp_name = extract_id_name_exp_name(B_path)
        A_id_B_exp_path = A_path.replace(A_exp_name, B_exp_name)
        A_id_B_exp_seg_path = A_seg_path.replace(A_exp_name, B_exp_name)
        B_id_A_exp_path = B_path.replace(B_exp_name, A_exp_name)
        B_id_A_exp_seg_path = B_seg_path.replace(B_exp_name, A_exp_name)
        if os.path.exists(A_path) and os.path.exists(B_path) and os.path.exists(A_id_B_exp_path) and os.path.exists(
                B_id_A_exp_path):
            try:
                should_flip = self.opt.isTrain and np.random.random() < 0.5
                res_dict = {
                    'real_A': self.get_concated_item_by_path(A_path, A_seg_path),
                    'path_A': A_path,
                    'real_B': self.get_concated_item_by_path(B_path, B_seg_path),
                    'path_B': B_path,
                    'real_A_id_B_exp': self.get_concated_item_by_path(A_id_B_exp_path, A_id_B_exp_seg_path),
                    'path_A_id_B_exp': A_id_B_exp_path,
                    'real_B_id_A_exp': self.get_concated_item_by_path(B_id_A_exp_path, B_id_A_exp_seg_path),
                    'path_B_id_A_exp': B_id_A_exp_path,

                    'id_name_A': A_id_name,
                    'exp_name_A': A_exp_name,
                    'id_name_B': B_id_name,
                    'exp_name_B': B_exp_name,
                }
                if should_flip:
                    for key in res_dict.keys():
                        if key.startswith('real_'):
                            res_dict[key] = F.hflip(res_dict[key])
                        elif key.startswith('exp_name_'):
                            res_dict[key] = FaceScapeBlendshape.reverse_exp_name(res_dict[key])
                return res_dict
            except Exception as e:
                print(A_id_name, A_exp_name, B_id_name, B_exp_name, e)
                return self.__getitem__(random.randint(0, len(self) - 1))
        else:
            return self.__getitem__(random.randint(0, len(self) - 1))

    def get_concated_item_by_path(self, A_path, B_path):
        try:
            if self.opt.replace_path is None:
                A_img = Image.fromarray(Memcached.cv2_imread(A_path))
            else:
                temp_A = Memcached.cv2_imread(A_path)
                replace_content = self.replace_img[:, :, 0]
                alpha = self.replace_img[:, :, 3] // 255
                temp_A = replace_content * alpha + temp_A * (1 - alpha)
                A_img = Image.fromarray(temp_A)
            A = self.transform_A(A_img)
            B_img = Image.fromarray(Memcached.cv2_imread(B_path))
            B = self.transform_B(B_img)
            return torch.cat([A, B], dim=0)
        except Exception as err:  # catch everything
            print(err)
            raise err
