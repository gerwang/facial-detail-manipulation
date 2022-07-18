import numpy as np

detail_bs_indices = np.array([0, 1, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 28, 29, 30, 31,
                              32, 33, 36, 37, 40, 41, 42, 43, 45, 48, 49, 50])


class BlendshapeReader:
    def __init__(self, input_path, use_fs_pro=False):
        self.use_fs_pro = use_fs_pro
        with open(input_path) as f:
            res = self.parse_file(f)
        self.bs = np.array(res)

    def parse_file(self, f):
        res = []
        for line in f.readlines():
            res.append(self.parse_line(line))
        return res

    def clip(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self.bs)
        self.bs = self.bs[start:end]

    def parse_line(self, line):
        line = line.split()
        index = 0
        while line[index] != 'C':
            index += 1
        index += 1
        num = int(line[index])
        index += 1
        res = list(map(float, line[index:index + num]))
        if self.use_fs_pro:
            res = np.array(res)
            res[2:4] = res[2:4] / 10
            res[[30, 31, 34, 35, 19]] = res[[30, 31, 34, 35, 19]] * 0.8
            res = res.tolist()

        return res


if __name__ == '__main__':
    input_path = '/home/gerw/Desktop/1230/data_processing/blendshape/clip_8.txt'
    reader = BlendshapeReader(input_path)
