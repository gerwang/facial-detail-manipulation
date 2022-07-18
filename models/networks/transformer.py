import torch

from models.networks import BaseNetwork
from models.networks.stylegan2_layers import EqualLinear
from util import util


class MappingNetwork(torch.nn.Module):
    def __init__(self, input_dim, cond_dim, output_dim, feature_dim, num_layers=8, lr_multiplier=0.01,
                 cond_in_all=False, latent_in=()):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.latent_in = latent_in
        self.cond_in_all = cond_in_all
        for i in range(self.num_layers):
            if i == 0:
                input_nc = input_dim
            else:
                input_nc = feature_dim
            if i != 0 and i in self.latent_in:
                input_nc += input_dim
            if i == 0 or self.cond_in_all:
                input_nc += cond_dim
            if i + 1 == self.num_layers:
                output_nc = output_dim
            else:
                output_nc = feature_dim
            layer = EqualLinear(input_nc, output_nc, lr_mul=lr_multiplier, activation='fused_lrelu')
            self.layers.append(layer)

    def forward(self, x, cond=None):
        x_flatten = x.flatten(start_dim=1)
        out = x_flatten
        for i, layer in enumerate(self.layers):
            if i != 0 and i in self.latent_in:
                out = torch.cat([out, x_flatten], dim=1)
            if cond is not None and (i == 0 or self.cond_in_all):
                out = torch.cat([out, cond], dim=1)
            out = layer(out)

        return out.view(x.shape)


class SpatialOnlyExpTransformer(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netTExp_feature_dim', type=int, default=2048)
        parser.add_argument('--netTExp_num_layers', type=int, default=8)
        parser.add_argument('--netTExp_lr_mult', type=float, default=0.01)
        parser.add_argument('--netTExp_use_cond_in_all', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--netTExp_latent_in', type=int, nargs='*', default=[])
        parser.add_argument('--netTExp_cond_dim', type=int, default=29)

        return parser

    def __init__(self, opt):
        super().__init__(opt)
        embedding_size = opt.crop_size // (2 ** opt.netE_num_downsampling_sp)
        self.embed_dim = opt.spatial_code_ch * embedding_size * embedding_size
        self.mapping = MappingNetwork(self.embed_dim, opt.netTExp_cond_dim, self.embed_dim,
                                      opt.netTExp_feature_dim, opt.netTExp_num_layers, opt.netTExp_lr_mult,
                                      opt.netTExp_use_cond_in_all, opt.netTExp_latent_in)

    def forward(self, spatial_code, global_code, cond):
        cond = cond[:, :self.opt.netTExp_cond_dim]
        return spatial_code + self.mapping(spatial_code, cond), global_code


class SpatialGlobalTransformer(SpatialOnlyExpTransformer):
    def __init__(self, opt):
        super().__init__(opt)
        self.global_embed_dim = opt.global_code_ch
        self.mapping_global = MappingNetwork(self.global_embed_dim, opt.netTExp_cond_dim, self.global_embed_dim,
                                             opt.netTExp_feature_dim, opt.netTExp_num_layers, opt.netTExp_lr_mult,
                                             opt.netTExp_use_cond_in_all, opt.netTExp_latent_in)

    def forward(self, spatial_code, global_code, cond):
        cond = cond[:, :self.opt.netTExp_cond_dim]
        return spatial_code + self.mapping(spatial_code, cond), global_code + self.mapping_global(global_code, cond)


class SpatialOnlyAgeTransformer(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netTAge_feature_dim', type=int, default=2048)
        parser.add_argument('--netTAge_num_layers', type=int, default=8)
        parser.add_argument('--netTAge_lr_mult', type=float, default=0.01)
        parser.add_argument('--netTAge_use_cond_in_all', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--netTAge_latent_in', type=int, nargs='*', default=[])
        parser.add_argument('--netTAge_cond_dim', type=int, default=1)

        return parser

    def __init__(self, opt):
        super().__init__(opt)
        embedding_size = opt.crop_size // (2 ** opt.netE_num_downsampling_sp)
        self.embed_dim = opt.spatial_code_ch * embedding_size * embedding_size
        self.mapping = MappingNetwork(self.embed_dim, opt.netTAge_cond_dim, self.embed_dim,
                                      opt.netTAge_feature_dim, opt.netTAge_num_layers, opt.netTAge_lr_mult,
                                      opt.netTAge_use_cond_in_all, opt.netTAge_latent_in)

    def forward(self, spatial_code, global_code, cond):
        cond = cond[:, -self.opt.netTAge_cond_dim:]
        return spatial_code + self.mapping(spatial_code, cond), global_code


class SpatialGlobalAgeTransformer(SpatialOnlyAgeTransformer):
    def __init__(self, opt):
        super().__init__(opt)
        self.global_embed_dim = opt.global_code_ch
        self.mapping_global = MappingNetwork(self.global_embed_dim, opt.netTAge_cond_dim, self.global_embed_dim,
                                             opt.netTAge_feature_dim, opt.netTAge_num_layers, opt.netTAge_lr_mult,
                                             opt.netTAge_use_cond_in_all, opt.netTAge_latent_in)

    def forward(self, spatial_code, global_code, cond):
        cond = cond[:, -self.opt.netTAge_cond_dim:]
        return spatial_code + self.mapping(spatial_code, cond), global_code + self.mapping_global(global_code, cond)


class ConcatGlobalTransformer(BaseNetwork):
    def forward(self, spatial_code, global_code, cond):
        return spatial_code, torch.cat([global_code, cond], dim=1)
