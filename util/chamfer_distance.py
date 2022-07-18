import numpy as np
from scipy.ndimage import distance_transform_edt


def get_seg(dpmap):
    """
    Args:
        dpmap: 256 x 256 uint16 already filtered
    Returns:
    """
    seg = np.zeros_like(dpmap, dtype=np.uint8)
    seg[dpmap < 30000] = 255
    return seg


def get_dt(seg):
    seg = np.logical_not(seg)
    dt = distance_transform_edt(seg)
    return dt


def chamfer_distance_one_side(seg, dt):
    return dt[seg != 0].sum() / (seg != 0).sum()


def chamfer_distance_dpmap_v1(input_dpmap, target_dpmap):
    input_seg = get_seg(input_dpmap)
    target_seg = get_seg(target_dpmap)
    input_dt = get_dt(input_seg)
    target_dt = get_dt(target_seg)
    return chamfer_distance_one_side(input_seg, target_dt) + chamfer_distance_one_side(target_seg, input_dt)


def chamfer_distance_dpmap_v2(input_dpmap, target_dpmap, sketch_gen):
    input_seg = sketch_gen.dpmap2seg(input_dpmap)
    target_seg = sketch_gen.dpmap2seg(target_dpmap)
    input_dt = get_dt(input_seg)
    target_dt = get_dt(target_seg)
    return chamfer_distance_one_side(input_seg, target_dt) + chamfer_distance_one_side(target_seg, input_dt)
