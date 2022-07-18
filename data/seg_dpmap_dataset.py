import os
import random

import PIL
from PIL.Image import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from natsort import natsorted
from scipy.ndimage import distance_transform_edt

import util
from data.base_dataset import BaseDataset
from util import extract_id_name_exp_name, load_clean_data_set
from util.memcache import Memcached
import torchvision.transforms.functional as F
from skimage.morphology import remove_small_objects, remove_small_holes
import cv2


class ToFloatTensor(object):
    def __call__(self, pic):
        if not isinstance(pic, np.ndarray):
            pic = np.array(pic)
        return torch.from_numpy(pic.astype(np.float32)[None, ...])

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNumChannel(object):
    def __init__(self, input_nc):
        self.input_nc = input_nc

    def __call__(self, pic):
        return pic.repeat((self.input_nc, 1, 1))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNumpy(object):
    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic.numpy()
        elif isinstance(pic, Image):
            return np.array(pic)
        elif isinstance(pic, np.ndarray):  # do nothing
            return pic
        else:
            raise ValueError(f'unknown type {type(pic)}')

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RemoveSmall(object):
    def __init__(self, enabled):
        self.enabled = enabled

    def __call__(self, seg):
        if self.enabled:
            point_size = (seg.shape[0] // 64) ** 2
            connectivity = seg.shape[0] // 128
            seg = remove_small_objects(seg != 0, point_size, connectivity=connectivity)
            seg = remove_small_holes(seg, point_size, connectivity=connectivity).astype(np.float32)
        return seg

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ResizeIfNeeded(object):
    def __init__(self, load_size, interpolation):
        self.load_size = load_size
        self.interpolation = interpolation

    def __call__(self, seg):
        if seg.shape[0] != self.load_size:
            seg = cv2.resize(seg, (self.load_size, self.load_size), interpolation=self.interpolation)
        return seg

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToDistanceMap(object):
    def __call__(self, img):
        img = np.logical_not(img)
        img = distance_transform_edt(img)
        img /= img.shape[0]
        return img


class ThresholdNormalize(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, img):
        img[img > self.threshold] = self.threshold
        img /= self.threshold
        return img


class Dilate(object):
    def __init__(self, dilation_size):
        self.element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                                 (dilation_size, dilation_size))

    def __call__(self, img):
        img = cv2.dilate(img, self.element)
        return img


class ToGray(object):
    def __call__(self, img):
        if img.ndim > 2:
            img = img[:, :, 0]
        return img


def get_seg_transforms(opt):
    return transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        ToNumpy(),
        RemoveSmall(opt.remove_small),
        Dilate(opt.dilate_size),
        transforms.ToTensor(),
        transforms.Normalize((opt.seg_mean,), (opt.seg_std,)),
        ToNumChannel(opt.input_nc),
    ])


def get_dpmap_transforms(opt):
    return transforms.Compose([
        ToNumpy(),
        ResizeIfNeeded(opt.load_size, cv2.INTER_AREA),
        ToFloatTensor(),
        transforms.Normalize((32768.0,), (32768.0,)),
        # transforms.RandomHorizontalFlip(),
        ToNumChannel(opt.input_nc),
    ])


def get_dt_transforms(opt):
    return transforms.Compose([
        ToNumpy(),
        ToGray(),
        RemoveSmall(opt.remove_small),
        ToDistanceMap(),
        ThresholdNormalize(opt.truncate_threshold),
        ResizeIfNeeded(opt.load_size, cv2.INTER_NEAREST),
        ToFloatTensor(),
        transforms.Normalize((opt.seg_mean,), (opt.seg_std,)),
        ToNumChannel(opt.input_nc),
    ])


class SegDpmapDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--data_type', choices=['seg', 'dpmap', 'dt'], default='seg')
        parser.add_argument('--dilate_size', type=int, default=0)
        parser.add_argument('--clean_data_path', default='./predef/clean_data.txt')
        parser.add_argument('--use_filtered', type=util.str2bool, const=True, default=False)
        parser.add_argument('--remove_small', type=util.str2bool, const=True, default=True)
        parser.add_argument('--truncate_threshold', type=float, default=0.05)
        parser.add_argument('--seg_mean', type=float, default=0.5)
        parser.add_argument('--seg_std', type=float, default=0.5)
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
        self.dir_A = opt.dataroot
        self.data_type = opt.data_type
        self.clean_data_set = load_clean_data_set(opt.clean_data_path)

        self.A_paths_dict, self.id_exp_name_dict = \
            self.make_dataset(self.dir_A, (self.data_type if self.data_type != 'dt' else 'mask_simp')
                              + ('_filtered' if opt.use_filtered else ''))
        if self.data_type == 'seg':
            self.transform_A = get_seg_transforms(opt)
        elif self.data_type == 'dpmap':
            self.transform_A = get_dpmap_transforms(opt)
        elif self.data_type == 'dt':
            self.transform_A = get_dt_transforms(opt)
        else:
            raise ValueError(f'unknown {self.data_type}')

    def __getitem__(self, index):
        A_path = self.A_paths_dict[self.current_phase][index % len(self)]
        return self.getitem_by_path(A_path)

    def getitem_by_path(self, A_path):
        try:
            A_img = PIL.Image.fromarray(Memcached.cv2_imread(A_path))
        except OSError as err:
            print(err)
            return self.__getitem__(random.randint(0, len(self) - 1))

        # apply image transformation
        A = self.transform_A(A_img)

        res_dict = {'real_A': A, 'path_A': A_path}
        if self.opt.isTrain and np.random.random() < 0.5:
            for key in res_dict.keys():
                if key.startswith('real_'):
                    res_dict[key] = F.hflip(res_dict[key])
        return res_dict

    def get_index_by_id_name_exp_name(self, id_name, exp_name):
        return self.id_exp_name_dict.get((id_name, exp_name), -1)

    def __len__(self):
        return len(self.A_paths_dict[self.current_phase])


if __name__ == '__main__':
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
