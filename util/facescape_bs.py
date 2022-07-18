import numpy as np
import torch


def get_bs_weight(full_bs=False):
    bs_weight = np.zeros((19, 51), dtype=np.float32)
    bs_weight[0, 30], bs_weight[0, 31] = 0.6, 0.6
    bs_weight[1, 19] = 0.7
    bs_weight[2, 14], bs_weight[2, 15], bs_weight[2, 36], bs_weight[2, 37] = 1, 1, 0.3, 0.3
    bs_weight[3, 21] = 1
    bs_weight[4, 22] = 1
    bs_weight[5, 23] = 1
    bs_weight[6, 42] = 1
    bs_weight[7, 43] = 1
    bs_weight[8, 32], bs_weight[8, 33] = 1, 1
    bs_weight[9, 29], bs_weight[9, 45] = 0.6, 1
    bs_weight[10, 40] = 1
    bs_weight[11, 41] = 1
    bs_weight[12, 14], bs_weight[12, 15], bs_weight[12, 36], bs_weight[12, 37] = 0.5, 0.5, 0.8, 0.8
    bs_weight[13, 19], bs_weight[13, 28], bs_weight[13, 29] = 0.23, 0.8, 0.6
    bs_weight[14, 30], bs_weight[14, 31], bs_weight[14, 45], bs_weight[14, 49], bs_weight[
        14, 50] = 0.5, 0.5, 1, 1, 1
    bs_weight[15, 48] = 1
    bs_weight[16, 0], bs_weight[16, 1] = 1, 1
    bs_weight[17, 16], bs_weight[17, 17], bs_weight[17, 18], bs_weight[17, 12], bs_weight[
        17, 13] = 0.4, 0.4, 0.4, 0.8, 0.8
    bs_weight[18, 14], bs_weight[18, 15] = 1, 1
    bs_weight = np.concatenate([np.zeros((1, 51), dtype=np.float32), bs_weight])  # 20 x 51
    if not full_bs:
        row_sum = bs_weight.sum(axis=0)
        bs_weight = bs_weight.T[np.where(row_sum != 0)].T  # 20 x 29
    bs_weight = np.ascontiguousarray(bs_weight)
    return bs_weight


class FaceScapeBlendshape:
    exp_list = ['1_neutral', '2_smile', '3_mouth_stretch', '4_anger', '5_jaw_left',
                '6_jaw_right', '7_jaw_forward', '8_mouth_left', '9_mouth_right', '10_dimpler',
                '11_chin_raiser', '12_lip_puckerer', '13_lip_funneler', '14_sadness', '15_lip_roll',
                '16_grin', '17_cheek_blowing', '18_eye_closed', '19_brow_raiser', '20_brow_lower']

    bs_weight = get_bs_weight()
    bs_mapping = {str(x): i for i, x in enumerate(bs_weight)}

    @staticmethod
    def reverse_exp_name(exp_name):
        if exp_name == '5_jaw_left':
            return '6_jaw_right'
        elif exp_name == '6_jaw_right':
            return '5_jaw_left'
        elif exp_name == '8_mouth_left':
            return '9_mouth_right'
        elif exp_name == '9_mouth_right':
            return '8_mouth_left'
        else:
            return exp_name

    @staticmethod
    def get_bs(exp_name):
        exp_idx = FaceScapeBlendshape.exp_list.index(exp_name)
        return FaceScapeBlendshape.bs_weight[exp_idx].copy()

    @staticmethod
    def get_bs_concat(exp_names):
        res = []
        for exp_name in exp_names:
            res.append(FaceScapeBlendshape.get_bs(exp_name)[None, ...])
        res = np.concatenate(res)
        return torch.from_numpy(res).float()

    @staticmethod
    def get_bs_cnt():
        return FaceScapeBlendshape.bs_weight.shape[1]

    @staticmethod
    def get_bs_name(bs):
        bs = bs[:FaceScapeBlendshape.get_bs_cnt()]
        for i in range(FaceScapeBlendshape.get_bs_cnt()):
            if np.all(FaceScapeBlendshape.bs_weight[i] == bs):
                return FaceScapeBlendshape.exp_list[i]
        return 'none'
