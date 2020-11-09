import numpy as np
from utils import singleton

@singleton
class FFTStatEncoder(object):

    def __init__(self):
        pass


    def encode(self, data):
        ret = []
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
