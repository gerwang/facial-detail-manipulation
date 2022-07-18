import numpy as np
import torch


class FaceScapeAge:
    def __init__(self, age_list_path):
        self.age_np = None
        self.load_age(age_list_path)

    def load_age(self, file_path):
        self.age_np = np.zeros(938, dtype=np.float32)
        with open(file_path) as fin:
            for line in fin.readlines():
                tmp = line.split()
                if len(tmp) >= 3:
                    num = int(tmp[0])
                    # num = self.index_map[num]
                    num -= 1
                    if tmp[2] == '-':
                        self.age_np[num] = -1
                    else:
                        self.age_np[num] = int(tmp[2])

    def normalize_age(self, age):
        return (age + np.random.randn() - 16) / (68 - 16)  # normalize to [0,1] same to blendshape

    def get_age(self, id_name):
        id_idx = int(id_name) - 1
        age = self.age_np[id_idx]
        if age != -1:
            age = self.normalize_age(age)
        else:
            age = np.random.uniform()
        return np.array([age], dtype=np.float32)

    def get_age_concat(self, id_names):
        res = []
        for id_name in id_names:
            res.append(self.get_age(id_name)[None, ...])
        res = np.concatenate(res)
        return torch.from_numpy(res).float()
