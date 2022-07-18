import os
import pickle
from os.path import join

import numpy as np
import openmesh as om

from util.facescape_bs import get_bs_weight, FaceScapeBlendshape
from copy import deepcopy
import logging


class OutputMeshBlendshapeKeyExp:
    def __init__(self, bilinear_model, predef_path):
        self.bilinear_model = bilinear_model
        self.template_mesh = om.read_trimesh(f'{predef_path}/convert_vt.obj', vertex_tex_coord=True)
        self.full_template_mesh = om.read_trimesh(f'{predef_path}/convert_vt_full.obj', vertex_tex_coord=True)
        self.indices = np.loadtxt(f'{predef_path}/front_vert_indices.txt', dtype=np.int32)
        self.bs_weight = get_bs_weight(full_bs=True)

    def process(self, root_path, param=None):
        if param is None:
            files = os.listdir(root_path)
            if 'params.npz' in files:
                param = np.load(join(root_path, 'params.npz'))
            elif 'large_scale_param.pkl' in files:
                param = pickle.load(open(join(root_path, 'large_scale_param.pkl'), 'rb'))
            else:
                logging.warning(f'Cannot find param file in {root_path}')
                return

        id_mat = self.bilinear_model.get_id_mat(param['id'])
        verts = self.bilinear_model.get_posed_face_from_id_mat(id_mat, param['exp'])

        template_mesh = deepcopy(self.template_mesh)
        template_mesh.points()[:] = verts[self.indices]
        om.write_mesh(join(root_path, 'current.obj'), template_mesh, vertex_tex_coord=True)

        full_template_mesh = deepcopy(self.full_template_mesh)
        full_template_mesh.points()[:] = verts
        om.write_mesh(join(root_path, 'full_large.obj'), full_template_mesh, vertex_tex_coord=True)

        os.makedirs(join(root_path, 'bs'), exist_ok=True)
        for i in range(len(param['exp'])):
            new_exp = np.zeros(len(param['exp']))
            new_exp[0] = 1
            new_exp[i] = 1
            verts = self.bilinear_model.get_posed_face_from_id_mat(id_mat, new_exp)
            self.template_mesh.points()[:] = verts[self.indices]
            om.write_mesh(join(root_path, 'bs', f'{i}.obj'), self.template_mesh, vertex_tex_coord=True)

        os.makedirs(f'{root_path}/key_exp', exist_ok=True)
        for i, exp_name in enumerate(FaceScapeBlendshape.exp_list):
            verts = self.bilinear_model.get_posed_face_from_id_mat(id_mat, np.concatenate(
                [np.ones(1), self.bs_weight[i]]))
            template_mesh.points()[:] = verts[self.indices]
            om.write_mesh(f'{root_path}/key_exp/{exp_name}.obj', template_mesh, vertex_tex_coord=True)
