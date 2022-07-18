import pickle

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize


class BilinearModel():
    def __init__(self, model_dir):
        self.num_lm = 68

        self.lm_index = np.zeros(self.num_lm, dtype=int)
        with open(f'{model_dir}/index_68.txt', 'r') as f:
            lines = f.readlines()
            for i in range(self.num_lm):
                line = lines[i]
                values = line.split()
                self.lm_index[i] = int(values[0])

        with open(f'{model_dir}/faces.pkl', 'rb') as f:
            self.texcoords, self.faces = pickle.load(f)
        with open(f'{model_dir}/front_verts_indices.pkl', 'rb') as f:
            self.front_verts_indices = pickle.load(f)
        with open(f'{model_dir}/front_texcoords.pkl', 'rb') as f:
            self.front_texcoords = pickle.load(f)
        with open(f'{model_dir}/front_faces.pkl', 'rb') as f:
            self.front_faces = pickle.load(f)

        with open(f'{model_dir}/id_mean.pkl', 'rb') as f:
            self.id_mean = pickle.load(f)
        with open(f'{model_dir}/id_var.pkl', 'rb') as f:
            self.id_var = pickle.load(f)
        with open(f'{model_dir}/exp_GMM.pkl', 'rb') as f:
            self.exp_gmm = pickle.load(f)
        with open(f'{model_dir}/front_faces.pkl', 'rb') as f:
            self.front_faces = pickle.load(f)

        with open(f'{model_dir}/contour_line_68.pkl', 'rb') as f:
            self.contour_line_right, self.contour_line_left = pickle.load(f)
        self.core_tensor = np.load(f'{model_dir}/core_847_50_52.npy')
        self.factors_id = np.load(f'{model_dir}/factors_id_847_50_52.npy')

        self.core_tensor = self.core_tensor.transpose((2, 1, 0))
        for i in range(51):
            self.core_tensor[:, i + 1, :] = self.core_tensor[:, i + 1, :] - self.core_tensor[:, 0, :]

        with open(f'{model_dir}/front_face_indices.pkl', 'rb') as f:
            self.front_face_indices = pickle.load(f)

        # for render
        tris = []
        self.vert_texcoords = np.zeros((len(self.front_verts_indices), 2))
        for face in self.front_faces:
            vertices, normals, texture_coords, material = face
            tris.append([vertices[0] - 1, vertices[1] - 1, vertices[2] - 1])
            for i in range(len(vertices)):
                self.vert_texcoords[vertices[i] - 1] = self.front_texcoords[texture_coords[i] - 1]
        self.tris = np.array(tris)

    def fit_image(self, img, lm_pos):
        w = img.shape[1]
        h = img.shape[0]
        lm_pos[:, 1] = h - lm_pos[:, 1]

        id = self.factors_id[0]
        exp = np.zeros(52)
        exp[0] = 1
        rot_vector = np.array([0, 0.1, 0], dtype=np.double)
        trans = np.array([0, 0])
        scale = 1

        mesh_vertices = self.core_tensor.dot(id).dot(exp).reshape((-1, 3))
        verts_img = self.project(mesh_vertices, rot_vector, scale, trans)
        lm_index = self.lm_index

        for optimize_loop in range(4):

            vertices_mean = np.mean(verts_img[lm_index], axis=0)
            vertices_2d = verts_img[lm_index] - vertices_mean
            lm_index_full = np.zeros(self.num_lm * 3, dtype=int)
            for i in range(self.num_lm * 3):
                lm_index_full[i] = lm_index[i // 3] * 3 + i % 3

            lm_mean = np.mean(lm_pos, axis=0)
            lm = lm_pos - lm_mean
            scale = np.sum(np.linalg.norm(lm, axis=1)) / np.sum(np.linalg.norm(vertices_2d, axis=1))
            trans = lm_mean - vertices_mean * scale

            lm_core_tensor = self.core_tensor[lm_index_full]

            lm_pos_3D = lm_core_tensor.dot(id).dot(exp).reshape((-1, 3))
            scale, trans, rot_vector = self._optimize_rigid_pos(scale, trans, rot_vector, lm_pos_3D, lm_pos)
            id = self._optimize_identity(scale, trans, rot_vector, id, exp, lm_core_tensor, lm_pos, prior_weight=1)
            exp = self._optimize_expression(scale, trans, rot_vector, id, exp, lm_core_tensor, lm_pos, prior_weight=1)

            mesh_vertices = self.core_tensor.dot(id).dot(exp).reshape((-1, 3))
            verts_img = self.project(mesh_vertices, rot_vector, scale, trans)

            lm_index = self._update_3d_lm_index(verts_img, lm_index)

        return {
            'rot_vector': rot_vector,
            'trans': trans,
            'scale': scale,
            'id': id,
            'exp': exp
        }

    def _update_3d_lm_index(self, points_proj, lm_index):
        updated_lm_index = list(lm_index)
        modify_key_right = range(9, 17)
        modify_key_left = range(0, 8)

        # get the outest point on the contour line
        for i in range(len(modify_key_right)):
            if len(self.contour_line_right[i]) != 0:
                max_ind = np.argmax(points_proj[self.contour_line_right[i], 0])
                updated_lm_index[modify_key_right[i]] = self.contour_line_right[i][max_ind]

        for i in range(len(modify_key_left)):
            if len(self.contour_line_left[i]) != 0:
                min_ind = np.argmin(points_proj[self.contour_line_left[i], 0])
                updated_lm_index[modify_key_left[i]] = self.contour_line_left[i][min_ind]

        bottom_cand = [11789, 1804, 11792, 5007, 11421, 1681, 11410, 5000, 11423, 3248, 11427, 1687, 15212, 6204, 15216,
                       2851]
        updated_lm_index[8] = bottom_cand[np.argmin((points_proj[bottom_cand, 1]))]

        return updated_lm_index

    def project(self, points, rot_vec, scale, trans, keepz=False):
        points_proj = self._rotate(points, rot_vec.reshape(1, 3))
        points_proj = points_proj * scale
        if keepz:
            points_proj[:, 0:2] = points_proj[:, 0:2] + trans
        else:
            points_proj = points_proj[:, 0:2] + trans
        return points_proj

    def compute_face(self, id, exp):
        return self.core_tensor.dot(id).dot(exp).reshape((-1, 3))

    def get_id_mat(self, id):
        return self.core_tensor.dot(id)

    @staticmethod
    def get_posed_face_from_id_mat(id_mat, exp):
        mesh_vertices = id_mat.dot(exp).reshape((-1, 3))
        return mesh_vertices

    def _rotate(self, points, rot_vec):
        """Rotate points by given rotation vectors.
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vec)
        with np.errstate(invalid='ignore'):
            v = rot_vec / theta
            v = np.nan_to_num(v)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + (points.dot(v.T) * (1 - cos_theta)).dot(v)

    def _optimize_rigid_pos(self, scale, trans, rot_vector, lm_pos_3D, lm_pos):
        lm_pos_3D = lm_pos_3D.ravel()
        lm_pos = lm_pos.ravel()
        params = np.hstack((scale, trans, rot_vector))
        result = least_squares(self._compute_res_rigid, params, verbose=0, x_scale='jac', ftol=1e-5, method='lm',
                               args=(lm_pos_3D, lm_pos))
        return result.x[0], result.x[1:3], result.x[3:6]

    def _compute_res_id(self, id, id_matrix, scale, trans, rot_vector, lm_pos, prior_weight):
        id_matrix = id_matrix.reshape(-1, id.shape[0])
        lm_pos_3D = id_matrix.dot(id).reshape((-1, 3))
        lm_proj = self.project(lm_pos_3D, rot_vector, scale, trans).ravel()
        return np.linalg.norm(lm_proj - lm_pos) ** 2 / scale / scale + prior_weight * (id - self.id_mean).dot(
            np.diag(1 / self.id_var)).dot(np.transpose([id - self.id_mean]))

    def _optimize_identity(self, scale, trans, rot_vector, id, exp, lm_core_tensor, lm_pos, prior_weight=20):
        id_matrix = np.tensordot(lm_core_tensor, exp, axes=([1], [0])).ravel()
        lm_pos = lm_pos.ravel()
        result = minimize(self._compute_res_id, id, method='L-BFGS-B',
                          args=(id_matrix, scale, trans, rot_vector, lm_pos, prior_weight), options={'maxiter': 100})
        return result.x

    def _compute_res_exp(self, exp, exp_matrix, scale, trans, rot_vector, lm_pos, prior_weight):
        exp_matrix = exp_matrix.reshape(-1, exp.shape[0] + 1)
        exp_full = np.ones(52)
        exp_full[1:52] = exp
        lm_pos_3D = exp_matrix.dot(exp_full).reshape((-1, 3))
        lm_proj = self.project(lm_pos_3D, rot_vector, scale, trans).ravel()

        return np.linalg.norm(lm_proj - lm_pos) ** 2 / scale / scale - prior_weight * \
               self.exp_gmm.score_samples(exp.reshape(1, -1))[0]

    def _optimize_expression(self, scale, trans, rot_vector, id, exp, lm_core_tensor, lm_pos, prior_weight=0.02):
        exp_matrix = np.dot(lm_core_tensor, id).ravel()
        lm_pos = lm_pos.ravel()
        bounds = []
        for i in range(exp.shape[0] - 1):
            bounds.append((0, 1))
        result = minimize(self._compute_res_exp, exp[1:52], method='L-BFGS-B', bounds=bounds,
                          args=(exp_matrix, scale, trans, rot_vector, lm_pos, prior_weight), options={'maxiter': 100})
        exp_full = np.ones(52)
        exp_full[1:52] = result.x
        return exp_full

    def _compute_res_rigid(self, params, lm_pos_3D, lm_pos):
        lm_pos_3D = lm_pos_3D.reshape(-1, 3)
        lm_proj = self.project(lm_pos_3D, params[3:6], params[0], params[1:3])
        return lm_proj.ravel() - lm_pos
