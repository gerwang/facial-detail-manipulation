import torch

from models import MultiGPUModelWrapper
from optimizers.swapping_autoencoder_optimizer import SwappingAutoencoderOptimizer
from util import interleave, has_arg, util
from util.facescape_age import FaceScapeAge
from util.facescape_bs import FaceScapeBlendshape


class SwappingAutoencoderReconCondOptimizer(SwappingAutoencoderOptimizer):
    """ Class for running the optimization of the model parameters.
    Implements Generator / Discriminator training, R1 gradient penalty,
    decaying learning rates, and reporting training progress.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        super(SwappingAutoencoderReconCondOptimizer, SwappingAutoencoderReconCondOptimizer).modify_commandline_options(
            parser, is_train)
        if not has_arg(parser, "--concat_age"):
            parser.add_argument("--concat_age", type=util.str2bool, const=True, default=False)
        if not has_arg(parser, '--age_list_path'):
            parser.add_argument('--age_list_path', default='./predef/info_list_v1.3.txt')
        return parser

    def __init__(self, model: MultiGPUModelWrapper):
        super().__init__(model)
        if self.opt.concat_age:
            self.facescape_age = FaceScapeAge(self.opt.age_list_path)

    def prepare_images(self, data_i):
        label_A = [FaceScapeBlendshape.get_bs_concat(data_i['exp_name_A'])]
        if self.opt.concat_age:
            label_A.append(self.facescape_age.get_age_concat(data_i['id_name_A']))
        label_A = torch.cat(label_A, dim=1).to(data_i['real_A'].device)

        label_B = [FaceScapeBlendshape.get_bs_concat(data_i['exp_name_B'])]
        if self.opt.concat_age:
            label_B.append(self.facescape_age.get_age_concat(data_i['id_name_B']))
        label_B = torch.cat(label_B, dim=1).to(data_i['real_B'].device)

        return interleave(data_i["real_A"], data_i['real_B']), \
               interleave(data_i['real_B_id_A_exp'], data_i['real_A_id_B_exp']), \
               interleave(label_A, label_B)
