import logging
import os

import cv2
from natsort import natsorted

from detail_shape_process.options import Options
from detail_shape_process.utils import filter_dpmap
from sketch.model_mse import SketchGenerator


class TrainingDetailProcessor:
    def __init__(self):
        self.opt = Options().parse()

        self.sketch_gen = SketchGenerator('checkpoints/model_mse.pth')

    def process(self):
        id_names = natsorted(os.listdir(self.opt.input))
        for id_name in id_names:
            logging.info(f'Processing {id_name}')
            os.makedirs(f'{self.opt.output}/{id_name}/dpmap', exist_ok=True)
            os.makedirs(f'{self.opt.output}/{id_name}/mask_simp_v6', exist_ok=True)

            for filename in natsorted(os.listdir(f'{self.opt.input}/{id_name}/dpmap')):
                exp_name, _ = os.path.splitext(filename)
                dpmap = cv2.imread(f'{self.opt.input}/{id_name}/dpmap/{filename}', cv2.IMREAD_UNCHANGED)
                dpmap = dpmap[600:2500, 1100:3000]
                dpmap = filter_dpmap(dpmap)
                cv2.imwrite(f'{self.opt.output}/{id_name}/dpmap/{filename}', dpmap)

                seg = self.sketch_gen.dpmap2seg(dpmap)
                cv2.imwrite(f'{self.opt.output}/{id_name}/mask_simp_v6/{filename}', seg)

            logging.info(f'Done {id_name}')


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    detail_processor = TrainingDetailProcessor()
    detail_processor.process()
