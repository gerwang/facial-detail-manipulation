import pickle
from cmd import Cmd
from time import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes

from detail_shape_process.camera import IntrinsicsCamera
from detail_shape_process.utils import get_device


def compute_vertex_normals(verts, faces):
    """Computes the packed version of vertex normals from the packed verts
    and faces. This assumes verts are shared between faces. The normal for
    a vertex is computed as the sum of the normals of all the faces it is
    part of weighed by the face areas.

    Args:
        refresh: Set to True to force recomputation of vertex normals.
            Default: False.
    """

    faces_packed = faces[0]
    verts_packed = verts[0]
    verts_normals = torch.zeros_like(verts_packed)
    vertices_faces = verts_packed[faces_packed]

    # NOTE: this is already applying the area weighting as the magnitude
    # of the cross product is 2 x area of the triangle.
    # pyre-fixme[16]: `Tensor` has no attribute `index_add`.
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 1],
        torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
            dim=1,
        ),
    )
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 2],
        torch.cross(
            vertices_faces[:, 0] - vertices_faces[:, 2],
            vertices_faces[:, 1] - vertices_faces[:, 2],
            dim=1,
        ),
    )
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 0],
        torch.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0],
            vertices_faces[:, 2] - vertices_faces[:, 0],
            dim=1,
        ),
    )

    _verts_normals_packed = torch.nn.functional.normalize(
        verts_normals, eps=1e-6, dim=1
    )
    return _verts_normals_packed


def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


# ---------------------------- process/generate vertices, normals, faces
def generate_triangles(h, w, margin_x=2, margin_y=5, mask=None):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    # .
    # w*h
    triangles = []
    for x in range(margin_x, w - 1 - margin_x):
        for y in range(margin_y, h - 1 - margin_y):
            triangle0 = [y * w + x, y * w + x + 1, (y + 1) * w + x]
            triangle1 = [y * w + x + 1, (y + 1) * w + x + 1, (y + 1) * w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:, [0, 2, 1]]
    return triangles


class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals


class SRenderY(nn.Module):
    def __init__(self, obj_filename, uv_size=256, rasterizer_type='pytorch3d'):
        super(SRenderY, self).__init__()
        self.uv_size = uv_size

        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        faces = faces.verts_idx[None, ...]

        if rasterizer_type == 'pytorch3d':
            self.uv_rasterizer = Pytorch3dRasterizer(uv_size)

        # faces
        dense_triangles = generate_triangles(uv_size, uv_size)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None, :, :])
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1;
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        # shape colors, for rendering shape overlay
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.
        face_colors = face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)

    def world2uv(self, vertices):
        '''
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        '''
        batch_size = vertices.shape[0]
        face_vertices_ = face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1), face_vertices_)[:, :3]
        return uv_vertices

    def extract(self, images, trans_verts):
        uv_pverts = self.world2uv(trans_verts)
        uv_gt = F.grid_sample(images, uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2], mode='bilinear', align_corners=True)
        return uv_gt

    def compute_uv_vis_map(self, verts, faces, dot_threshold):
        normals = compute_vertex_normals(verts, faces)
        if dot_threshold < 0:
            tmp = normals[:, 2] < dot_threshold
        else:
            tmp = normals[:, 2] > dot_threshold
        visibility = tmp.float()[..., None].repeat((1, 3))[None, ...]
        batch_size = 1
        face_vertices_ = face_vertices(visibility, self.faces.expand(batch_size, -1, -1))
        uv_vis_map = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                        self.uvfaces.expand(batch_size, -1, -1), face_vertices_)[:, :3]
        uv_vis_map[uv_vis_map > 0.5] = 1
        uv_vis_map[uv_vis_map <= 0.5] = 0
        return uv_vis_map.cpu().numpy()[0].transpose([1, 2, 0]) * 255

    def extract_perspective(self, image_path, mesh_path, param_path):
        start_time = time()
        image = np.ascontiguousarray(cv2.imread(image_path)[::-1])
        image = torch.from_numpy(image).float().permute([2, 0, 1]).unsqueeze(0).cuda()
        verts, faces, _ = load_obj(mesh_path, device='cuda')
        param = pickle.load(open(param_path, 'rb'))
        K = param['K']
        camera = IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
        K_th = torch.from_numpy(camera.get_projection_matrix(image.shape[3], image.shape[2])).float().cuda()
        cv2gl = torch.tensor([
            [1, 0, 0, ],
            [0, -1, 0],
            [0, 0, -1],
        ]).float().cuda()
        scale = param['scale']
        rmat, _ = cv2.Rodrigues(param['rot_vector'])
        rmat = torch.from_numpy(rmat).float().cuda()
        tvec = torch.from_numpy(param['trans']).float().cuda()
        verts, faces, _ = load_obj(mesh_path, device='cuda')
        verts = verts * scale @ rmat.t().unsqueeze(0) + tvec.unsqueeze(0)
        # camera space
        uv_vis_map = self.compute_uv_vis_map(verts, faces, -0.5)
        verts = verts @ cv2gl.t().unsqueeze(0)
        verts = torch.cat([verts, verts.new_ones(verts.shape[:-1] + (1,))], dim=-1)
        verts = verts @ K_th.t().unsqueeze(0)
        verts = verts[:, :, :3] / verts[:, :, 3:]
        res = self.extract(image, verts)
        res = res[0].cpu().numpy().transpose([1, 2, 0])
        print(f'Process one {time() - start_time}')
        return res, uv_vis_map

    def extract_weak_perspective(self, image_path, mesh_path, param_path):
        image = np.ascontiguousarray(cv2.imread(image_path)[::-1])
        image = torch.from_numpy(image).float().permute([2, 0, 1]).unsqueeze(0).cuda()
        verts, faces, _ = load_obj(mesh_path, device='cuda')
        param = np.load(param_path)
        scale = float(param['scale'])
        rmat, _ = cv2.Rodrigues(param['rot_vector'])
        rmat = torch.from_numpy(rmat).float().cuda()
        tvec = torch.from_numpy(param['trans']).float().cuda()
        verts = verts * scale @ rmat.t().unsqueeze(0)
        verts[:, :, :2] += tvec[None, None, ...]
        # camera_space
        uv_vis_map = self.compute_uv_vis_map(verts, faces, 0.5)
        # uv_vis_map = self.compute_uv_vis_map(verts, faces, 0.15)  # prev value: 0.5
        half_size = torch.tensor([image.shape[3] / 2, image.shape[2] / 2]).float().cuda()
        verts[:, :, :2] -= half_size[None, None, ...]
        verts[:, :, :2] /= half_size[None, None, ...]
        res = self.extract(image, verts)
        res = res[0].cpu().numpy().transpose([1, 2, 0])
        return res, uv_vis_map


class TextureExtractor(Cmd):
    prompt = '> '

    def __init__(self, predef_path='./predef', out_size=1024, gpu_ids=-1):
        super().__init__()
        self.render = SRenderY(
            f'{predef_path}/convert_vt_full.obj',
            out_size).to(get_device(gpu_ids))

    def do_extract(self, this_path):
        try:
            res, mask = self.render.extract_perspective(f'{this_path}/undistort_img.png',
                                                        f'{this_path}/full_large.obj',
                                                        f'{this_path}/large_scale_param.pkl')
            cv2.imwrite(f'{this_path}/texture.png',
                        res)
            cv2.imwrite(f'{this_path}/mask.png',
                        mask)
            print(f'Done {this_path}')
        except Exception as e:
            print(this_path, e)

    @staticmethod
    def inpaint_sym(tex, mask):
        mask = mask[..., 0].astype(np.uint8)
        row_idx, col_idx = np.nonzero(255 - mask)
        sym_row_idx, sym_col_idx = row_idx, tex.shape[1] - 1 - col_idx
        valid = mask[sym_row_idx, sym_col_idx] != 0
        res = tex.copy()
        res[row_idx[valid], col_idx[valid]] = tex[sym_row_idx[valid], sym_col_idx[valid]]
        return res

    def do_extract_wp(self, this_path):
        try:
            res, mask = self.render.extract_weak_perspective(f'{this_path}/undistort_img.png',
                                                             f'{this_path}/full_large.obj',
                                                             f'{this_path}/params.npz')
            res = self.inpaint_sym(res, mask)
            cv2.imwrite(f'{this_path}/texture.png',
                        res)
            cv2.imwrite(f'{this_path}/mask.png',
                        mask)
        except Exception as e:
            print(this_path, e)
            raise e


def main():
    texture_extractor = TextureExtractor()
    texture_extractor.cmdloop()
