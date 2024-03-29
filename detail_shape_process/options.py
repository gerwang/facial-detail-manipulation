import argparse
import torch


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='dpmap_single',
                                 help='"dpmap_single":predicting the displacemnt map for the source image. "dpmap_rig":predicting the displacemnt map for 20 key expressions.')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                                 help='pretrained models are saved here')
        self.parser.add_argument('--predef_dir', type=str, default='./predef', help='predefined files are saved here')
        self.parser.add_argument('--input', type=str, help='input dir', default='./demo/input')
        self.parser.add_argument('--output', type=str, help='output dir', default='./demo/output')
        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--total_workers', type=int, default=1)
        self.parser.add_argument('--worker_idx', type=int, default=0)

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        return self.opt
