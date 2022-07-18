import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms


class SketchGenerator:
    def __init__(self, ckpt_path):
        self.model = nn.Sequential(  # Sequential,
            nn.Conv2d(1, 48, (5, 5), (2, 2), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(48, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, (4, 4), (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 48, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 48, (4, 4), (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.Conv2d(48, 24, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(24, 1, (3, 3), (1, 1), (1, 1)),
            nn.Sigmoid(),
        )

        self.immean = 0.9664423107454593
        self.imstd = 0.08583666033640507
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.model = self.model.cuda()

        self.denoise_strength = 20
        self.scale_factor = 0.23

    def dpmap2seg(self, dpmap):
        """
        :param dpmap: (256, 256) uint16
        :return: (256, 256) uint8
        """
        dpmap = (dpmap.astype(np.float32) - 32768) / 32768
        # print(np.mean(dpmap), np.std(dpmap))
        # dpmap = (dpmap - np.mean(dpmap)) / np.std(dpmap)  # convert to N(0,1)
        dpmap = dpmap / np.abs(np.mean(dpmap[dpmap < 0]))  # convert to N(0,1)
        # dpmap = cv2.GaussianBlur(dpmap, (3, 3), 1)
        dpmap = 1 - np.clip(-dpmap * self.scale_factor, a_min=0, a_max=1)
        dpmap = (dpmap * 255).astype(np.uint8)
        dpmap = cv2.fastNlMeansDenoising(dpmap, None, self.denoise_strength, 7, 21)
        data = Image.fromarray(dpmap)
        w, h = data.size[0], data.size[1]
        pw = 8 - (w % 8) if w % 8 != 0 else 0
        ph = 8 - (h % 8) if h % 8 != 0 else 0
        data = ((transforms.ToTensor()(data) - self.immean) / self.imstd).unsqueeze(0)
        if pw != 0 or ph != 0:
            data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data)

        if self.use_cuda:
            data = data.cuda()
        pred = self.model(data)
        pred = pred.detach().cpu().numpy()[0, 0]
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        pred = 1 - pred
        pred = (pred * 255).astype(np.uint8)
        return pred
