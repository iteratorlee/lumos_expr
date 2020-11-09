import numpy as np
from conf import LumosConf
from utils import singleton

@singleton
class FFTStatEncoder(object):

    def __init__(self):
        pass


    def encode(self, data):
        ret = []
        conf = LumosConf()
        valid_idx = conf.get('dataset', 'valid_idx')
        if data.shape[1] != len(valid_idx):
            data = data[:, valid_idx]
        print(data.shape)
        for i in range(data.shape[1]):
            tmp = []
            series = data[:, i]
            fft_feat = self.__fft(series)
            stat_feat = self.__stat(series)
            tmp.extend(fft_feat)
            tmp.extend(stat_feat)
            ret.extend(tmp)
        return ret


    def __fft(self, series):
        return [1, 2]


    def __stat(self, series):
        return [3, 4]
