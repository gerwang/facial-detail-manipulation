import torch

from models.networks import BaseNetwork
from models.networks.stylegan2_layers import Discriminator as OriginalStyleGAN2Discriminator
from util.facescape_bs import FaceScapeBlendshape


class StyleGAN2Discriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--netD_scale_capacity", default=1.0, type=float)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.stylegan2_D = OriginalStyleGAN2Discriminator(
            opt.crop_size,
            2.0 * opt.netD_scale_capacity,
            blur_kernel=[1, 3, 3, 1] if self.opt.use_antialias else [1],
            input_nc=opt.input_nc,
        )

    def forward(self, x, extract_features=False):
        pred = self.stylegan2_D(x, extract_features)
        return pred

    def get_features(self, x):
        return self.stylegan2_D.get_features(x)

    def get_pred_from_features(self, feat, label):
        assert label is None
        feat = feat.flatten(1)
        out = self.stylegan2_D.final_linear(feat)
        return out


class StyleGAN2HalfInDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--netD_scale_capacity", default=1.0, type=float)
        parser.add_argument('--netD_input_channel_index', default=1, type=int)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.stylegan2_D = OriginalStyleGAN2Discriminator(
            opt.crop_size,
            2.0 * opt.netD_scale_capacity,
            blur_kernel=[1, 3, 3, 1] if self.opt.use_antialias else [1],
            input_nc=1,
        )
        self.input_index = self.opt.netD_input_channel_index

    def forward(self, x, extract_features=False):
        pred = self.stylegan2_D(x[:, self.input_index:self.input_index + 1],
                                extract_features)
        return pred


class StyleGAN2AuxOutDiscriminator(StyleGAN2Discriminator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        super(StyleGAN2AuxOutDiscriminator, StyleGAN2AuxOutDiscriminator).modify_commandline_options(parser, is_train)
        parser.add_argument('--netD_aux_output_dim', type=int, default=29)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.stylegan2_D = OriginalStyleGAN2Discriminator(
            opt.crop_size,
            2.0 * opt.netD_scale_capacity,
            blur_kernel=[1, 3, 3, 1] if self.opt.use_antialias else [1],
            input_nc=opt.input_nc,
            output_nc=1 + opt.netD_aux_output_dim,
        )

    def forward(self, x):
        pred = self.stylegan2_D(x)
        pred, aux_out = torch.split(pred, [1, self.opt.netD_aux_output_dim], dim=1)
        return pred, aux_out


class StyleGAN2MultiHeadDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--netD_scale_capacity", default=1.0, type=float)
        parser.add_argument('--netD_exp_clusters', type=int, default=20)
        parser.add_argument('--netD_age_clusters', type=int, default=7)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.stylegan2_D = OriginalStyleGAN2Discriminator(
            opt.crop_size,
            2.0 * opt.netD_scale_capacity,
            blur_kernel=[1, 3, 3, 1] if self.opt.use_antialias else [1],
            input_nc=opt.input_nc,
            output_nc=opt.netD_exp_clusters + opt.netD_age_clusters,
        )
        self.register_buffer('age_cluster', torch.tensor(
            [(i + 0.5) / opt.netD_age_clusters for i in range(opt.netD_age_clusters)]).float())

    def forward(self, x, label, extract_features=False):
        if extract_features:
            pred, feat = self.stylegan2_D(x, extract_features)  # (B, exp+age)
        else:
            pred = self.stylegan2_D(x, extract_features)  # (B, exp+age)
        exp_label = label[:, :FaceScapeBlendshape.get_bs_cnt()]
        age_label = label[:, FaceScapeBlendshape.get_bs_cnt():]
        pred_exp = pred[:, :self.opt.netD_exp_clusters]
        pred_age = pred[:, self.opt.netD_exp_clusters:]
        exp_idx = torch.tensor([FaceScapeBlendshape.bs_mapping[str(x)] for x in exp_label.detach().cpu().numpy()],
                               dtype=torch.long, device=exp_label.device)
        age_idx = torch.argmin((age_label - self.age_cluster[None, ...]).abs(), dim=1)
        range_idx = torch.arange(pred.shape[0], device=pred.device)
        pred_exp = pred_exp[range_idx, exp_idx]
        pred_age = pred_age[range_idx, age_idx]
        if extract_features:
            return pred_exp, pred_age, feat
        else:
            return pred_exp, pred_age
