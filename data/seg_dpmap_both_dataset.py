import copy
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from natsort import natsorted

import util
from data.base_dataset import BaseDataset
from data.seg_dpmap_dataset import get_seg_transforms, get_dpmap_transforms, SegDpmapDataset, get_dt_transforms
from util import extract_id_name_exp_name, load_clean_data_set
from util.memcache import Memcached


class SegDpmapBothDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--seg_type', choices=['seg', 'dt'], default='seg')
        parser.add_argument('--seg_dataroot')
        parser.add_argument('--dilate_size', type=int, default=0)
        parser.add_argument('--clean_data_path', default='./predef/clean_data.txt')
        parser.add_argument('--use_mask_simp', type=util.str2bool, const=True, default=True)
        parser.add_argument('--use_filtered', type=util.str2bool, const=True, default=False)
        parser.add_argument('--remove_small', type=util.str2bool, const=True, default=True)
        parser.add_argument('--truncate_threshold', type=float, default=0.05)
        parser.add_argument('--dir_A_name')
        parser.add_argument('--seg_mean', type=float, default=0)
        parser.add_argument('--seg_std', type=float, default=1)
        return parser

    def make_dataset(self, dir, data_type):
        res = {
            'train': [],
            'val': [],
            'test': [],
        }
        id_exp_name_dict = {}
        for id_name in natsorted(os.listdir(dir)):
            id_idx = int(id_name)
            if id_idx % 10 == 0:
                phase = 'test'
            else:
                phase = 'train'
            if os.path.isdir(f'{dir}/{id_name}'):
                for file_name in natsorted(os.listdir(f'{dir}/{id_name}/{data_type}')):
                    file_path = f'{dir}/{id_name}/{data_type}/{file_name}'
                    id_name, exp_name = extract_id_name_exp_name(file_path)
                    if (id_name, exp_name) not in self.clean_data_set:  # dirty data
                        continue
                    id_exp_name_dict[(id_name, exp_name)] = len(res[phase])
                    res[phase].append(file_path)
        for key in res.keys():
            res[key] = list(natsorted(res[key]))
        res['val'] = res['test']  # trick to make val same as test, the most common mistake in ML
        return res, id_exp_name_dict

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_B = opt.dataroot
        if opt.seg_dataroot is None:
            self.dir_A = self.dir_B
        else:
            self.dir_A = opt.seg_dataroot
        self.clean_data_set = load_clean_data_set(opt.clean_data_path)

        if opt.dir_A_name is None:
            opt.dir_A_name = 'mask_simp' if opt.use_mask_simp else 'seg'
        self.A_paths_dict, self.id_exp_name_dict = self.make_dataset(self.dir_A, opt.dir_A_name)
        self.B_paths_dict, _ = self.make_dataset(self.dir_B, 'dpmap' + ('_filtered' if opt.use_filtered else ''))
        assert len(self.A_paths_dict[self.current_phase]) == len(self.B_paths_dict[self.current_phase])
        old_input_nc = opt.input_nc
        opt.input_nc //= 2
        if opt.seg_type == 'seg':
            self.transform_A = get_seg_transforms(opt)
        elif opt.seg_type == 'dt':
            self.transform_A = get_dt_transforms(opt)
        else:
            raise ValueError(f'unknown {opt.seg_type}')
        self.transform_B = get_dpmap_transforms(opt)
        opt.input_nc = old_input_nc

    def __getitem__(self, index):
        should_flip = self.opt.isTrain and np.random.random() < 0.5
        A_paths = self.A_paths_dict[self.current_phase]
        A_path = A_paths[index % len(A_paths)]
        A_dict = self.getitem_by_path(A_path, self.transform_A, should_flip)
        B_paths = self.B_paths_dict[self.current_phase]
        B_path = B_paths[index % len(B_paths)]
        B_dict = self.getitem_by_path(B_path, self.transform_B, should_flip)
        A_dict['real_A'] = torch.cat([A_dict['real_A'], B_dict['real_A']], dim=0)
        return A_dict

    def getitem_by_path(self, A_path, transform, should_flip, raise_error=False):
        try:
            A_img = Image.fromarray(Memcached.cv2_imread(A_path))
        except (OSError, AttributeError) as err:
            if raise_error:
                raise err
            print(err)
            return self.__getitem__(random.randint(0, len(self) - 1))

        # apply image transformation
        A = transform(A_img)

        res_dict = {'real_A': A, 'path_A': A_path}
        if should_flip:
            for key in res_dict.keys():
                if key.startswith('real_'):
                    res_dict[key] = F.hflip(res_dict[key])
        return res_dict

    def get_index_by_id_name_exp_name(self, id_name, exp_name):
        return self.id_exp_name_dict.get((id_name, exp_name), -1)

    def __len__(self):
        return len(self.A_paths_dict[self.current_phase])


if __name__ == '__main__':
    import cv2
    from easydict import EasyDict

    opt = EasyDict(dataroot='./predef/fs1024_256_filtered/',
                   data_type='dpmap')
    dataset = SegDpmapDataset(opt)
    print(len(dataset))
    tmp = dataset[0]
    # print(tmp)
    print(tmp['real_A'].max(), tmp['real_A'].min())
    print(tmp['path_A'])
    a = tmp['real_A'].numpy().transpose([1, 2, 0])
    cv2.imshow('frame', (a + 1) * 0.5)
    cv2.waitKey(0)
