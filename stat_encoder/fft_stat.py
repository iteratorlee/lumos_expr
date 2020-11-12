import numpy as np
from conf import LumosConf
from utils import singleton
from scipy.fftpack import fft, ifft

@singleton
class FFTStatEncoder(object):

    def __init__(self):
        pass


    def encode(self, norm_data, raw_data):
        ret = []
        conf = LumosConf()
        valid_idx = conf.get('dataset', 'valid_idx')
        if norm_data.shape[1] != len(valid_idx):
            norm_data = norm_data[:, valid_idx]
        if raw_data.shape[1] != len(valid_idx):
            raw_data = raw_data[:, valid_idx]
        for i in range(norm_data.shape[1]):
            tmp = []
            norm_series = norm_data[:, i]
            raw_series = raw_data[:, i]
            fft_feat = self.__fft(norm_series)
            stat_feat = self.__stat(raw_series, i)
            tmp.extend(fft_feat)
            tmp.extend(stat_feat)
            ret.extend(tmp)
        return ret


    def __fft(self, norm_series, n_feat=2, sample_interval=5):
        len_s = len(norm_series)
        N = int(np.power(2, np.ceil(np.log2(len_s))))
        fft_y = fft(norm_series, N)[:N // 2] / len_s * 2
        fft_y_abs = np.abs(fft_y)
        fre = np.arange(N // 2) / N * sample_interval
        top_amp = np.sort(fft_y_abs)[-n_feat:]
        top_idx = np.argsort(fft_y_abs)[-n_feat:]
        top_fre = fre[top_idx]
        return list(top_amp) + list(top_fre)


    def __stat(self, raw_series, idx):
        conf = LumosConf()
        valid_max_val = conf.get('dataset', 'valid_max_vals')[idx]
        max_val = np.max(raw_series) / valid_max_val
        min_val = np.min(raw_series) / valid_max_val
        avg_val = np.mean(raw_series) / valid_max_val
        # var_val = np.var(raw_series) / valid_max_val
        return [max_val, min_val, avg_val]
