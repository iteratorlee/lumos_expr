import numpy as np
from conf import LumosConf
from utils import singleton
from scipy.fftpack import fft, ifft
from data_loader import DataLoader

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


    def __fft(self, series, n_feat=2, sample_interval=5):
        len_s = len(series)
        N = int(np.power(2, np.ceil(np.log2(len_s))))
        fft_y = fft(series, N)[:N // 2] / len_s * 2
        fft_y_abs = np.abs(fft_y)
        fre = np.arange(N // 2) / N * sample_interval
        top_amp = np.sort(fft_y_abs)[-n_feat:]
        top_idx = np.argsort(fft_y_abs)[-n_feat:]
        top_fre = fre[top_idx]
        return list(top_amp) + list(top_fre)


    def __stat(self, series):
        return [3, 4]
