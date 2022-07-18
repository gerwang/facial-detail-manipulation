import logging
import os

import cv2
import numpy as np
import openmesh as om
import torch
import torchvision.transforms.functional as F
from PIL import Image
from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor
from natsort import natsorted

from detail_shape_process.bilinear_model import BilinearModel
from detail_shape_process.options import Options
from detail_shape_process.output_bs import OutputMeshBlendshapeKeyExp
from detail_shape_process.pix2pixHD.models import create_model
from detail_shape_process.texture_extractor import TextureExtractor
from detail_shape_process.utils import color_transfer, tensor2im, get_device, preserve_largest_area, filter_dpmap


class DetailProcessor:
    def __init__(self):
        self.opt = Options().parse()

        logging.info(f'Init bilinear model')
        self.bilinear_model = BilinearModel(self.opt.predef_dir)
        self.bs_template_mesh = om.read_trimesh(f'{self.opt.predef_dir}/convert_vt.obj', vertex_tex_coord=True)

        logging.info(f'Init landmark detector')
        alignment_device = get_device(self.opt.gpu_ids)
        face_detector_class = (RetinaFacePredictor, 'RetinaFace')
        fd_model = face_detector_class[0].get_model()
        self.face_detector = face_detector_class[0](
            threshold=0.8, device=alignment_device, model=fd_model)
        fa_model = FANPredictor.get_model()
        fa_config = FANPredictor.create_config(use_jit=True)
        self.landmark_detector = FANPredictor(device=alignment_device, model=fa_model, config=fa_config)

        logging.info(f'Init dpmap model')
        self.dpmap_model = create_model(self.opt)
        self.mask = (255 - cv2.imread(f'{self.opt.predef_dir}/front_mask.png')[:, :, 0]).astype(bool)

        self.output_bs = OutputMeshBlendshapeKeyExp(self.bilinear_model, self.opt.predef_dir)

        logging.info(f'Init texture extractor')
        self.texture_extractor = TextureExtractor(self.opt.predef_dir, gpu_ids=self.opt.gpu_ids)

    def get_img_names(self):
        img_names = []
        img_idx = 0
        for name in natsorted(os.listdir(self.opt.input)):
            if any(name.endswith(extension) for extension in
                   ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff']):
                if img_idx % self.opt.total_workers == self.opt.worker_idx:
                    img_names.append(name)
                img_idx += 1
        return img_names

    def detect_landmark(self, image, max_size=1080):
        """
        image: 0-255, uint8, bgr, (h, w, 3)
        max_size: larger size image will be downsampled
        return: detected landmarks (68, 2)
        """
        factor = 1.0
        while image.shape[1] > max_size:
            # downsample the image since the landmark detector performs worse when using large (e.g. 4K) images
            image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)
            factor *= 2.0
        faces = self.face_detector(image, rgb=False)
        # If there are multiple faces detected, preserve the face with the largest area
        faces = preserve_largest_area(faces)
        if len(faces) == 0:
            return None
        else:
            landmarks, scores = self.landmark_detector(image, faces, rgb=False)
            kpt = landmarks[0]
            kpt *= factor
            return kpt

    def gen_kep_exp_blendshape(self, this_output_path, save_dict):
        self.output_bs.process(this_output_path, save_dict)

    def gen_dpmap(self, texture_path):
        texture = cv2.imread(texture_path)
        new_pixels = color_transfer(texture[self.mask][:, np.newaxis, :])
        texture[:] = 0
        texture[self.mask] = new_pixels[:, 0, :]
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB).astype(np.float32)
        texture = np.transpose(texture, (2, 0, 1))
        texture = torch.tensor(texture) / 255
        texture = F.normalize(texture, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), True)
        texture = torch.unsqueeze(texture, 0)

        with torch.no_grad():
            dpmap = self.dpmap_model.inference(texture, torch.tensor(0))
            dpmap = tensor2im(dpmap.detach()[0])
            dpmap = filter_dpmap(dpmap)
        return dpmap

    def process(self):
        img_names = self.get_img_names()
        for img_name in img_names:
            logging.info(f'Processing {img_name}')

            base_name = os.path.splitext(img_name)[0]
            if not os.path.exists(f'{self.opt.output}/{base_name}'):
                os.makedirs(f'{self.opt.output}/{base_name}')
            img = cv2.imread(f'{self.opt.input}/{img_name}')
            lm_pos = self.detect_landmark(img)
            if lm_pos is None:
                logging.warning(f'Cannot detect faces, skipping {img_name}')
                return

            logging.info('Fitting 3DMM Parameters...')

            save_dict = self.bilinear_model.fit_image(img, lm_pos)
            np.savez(f'{self.opt.output}/{base_name}/params.npz', **save_dict)

            logging.info('Generating blendshapes...')

            self.gen_kep_exp_blendshape(f'{self.opt.output}/{base_name}', save_dict)

            logging.info('Unwrap texture...')

            cv2.imwrite(f'{self.opt.output}/{base_name}/undistort_img.png', img)
            self.texture_extractor.do_extract_wp(f'{self.opt.output}/{base_name}')

            logging.info('Reconstructing displacement maps...')
            dpmap = self.gen_dpmap(f'{self.opt.output}/{base_name}/texture.png')
            Image.fromarray(dpmap).save(f'{self.opt.output}/{base_name}/dpmap.png')

            logging.info(f'Done {img_name}')


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    detail_processor = DetailProcessor()
    detail_processor.process()
