import os
import sys
import json
import time
import numpy as np

from collections import defaultdict

from conf import LumosConf
from data_loader import DataLoader

from third_party.keras_lstm_vae.lstm_vae import create_lstm_vae


class TopEncoder(object):
    '''
    every workload should have a LSTM autoencoder, which is trained seperated from others
    '''
    def __init__(self, to_load_workloads=1):
        '''
        params:
        @to_load_workloads
            type: int/list
            the encoders that needs to be loaded
        '''
        self.to_load_workloads = to_load_workloads
        self.load_encoders()


    def load_encoders(self):
        '''
        TODO: load_encoders from TF dump files
        '''
        pass

        
    def encode_metrics(self, metrics):
        '''
        TODO: encode the metrics
        '''
        return metrics


if __name__ == "__main__":
    conf = LumosConf()
    dump_pth = conf.get('dataset', 'dump_pth')
    data_loader = DataLoader(dump_pth=dump_pth)
    data_loader.load_data()
    data = data_loader.get_data()
    
    samples4enc = get_samples(data)
    for wl in samples4enc:
        samples4enc[wl] = padding_data(samples4enc[wl])

    sampels4lumos = get_left_samples(data)
    for wl in sampels4lumos:
        sampels4lumos[wl] = padding_data(sampels4lumos[wl])
