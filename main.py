import os
import sys
import json
import time
import numpy as np

from keras.regularizers import l1
from keras.layers import Dense, Input
from keras.models import Sequential, load_model
from utils import *

from conf import LumosConf
from model import LumosModel
from data_loader import DataLoader
from third_party.keras_lstm_vae.lstm_vae import create_lstm_vae


if __name__ == "__main__":
    conf = LumosConf()
    dump_pth = conf.get('dataset', 'dump_pth')
    data_loader = DataLoader(dump_pth=dump_pth)
    data_loader.load_data()
    data = data_loader.get_data()
    
    samples4enc = get_samples(data)
    samples4lumos = get_left_samples(data)

    max_lens_enc = get_max_lens(samples4enc)
    max_lens_lumos = get_max_lens(samples4lumos)

    max_lens = {}
    for wl in max_lens_enc:
        max_lens[wl] = max(max_lens_enc[wl], max_lens_lumos[wl])
    
    # samples4enc is used to train the encoder model
    # samples4lumos is used to train the prediction model
    
    for wl in samples4enc:
        padding_data(samples4enc[wl], max_len=max_lens[wl])
    
    for wl in samples4lumos:
        padding_data(samples4lumos[wl], max_len=max_lens[wl])
    
    sample_metrics4enc = defaultdict(lambda: [])
    for wl, _data in samples4enc.items():
        sample_metrics4enc[wl].extend([ele.get_metrics() for ele in _data])

    vae_dict = {}
    test_wls = ['hadoop_aggregation'] # for debug
    # for wl, in sample_metrics4enc:
    for wl in test_wls:
        wl_data = sample_metrics4enc[wl]
        vae, enc, gen = create_lstm_vae(
            input_dim=wl_data[0].shape[1],
            timesteps=wl_data[0].shape[0],
            batch_size=conf.get('encoder', 'batch_size'),
            intermediate_dim=conf.get('encoder', 'intermediate_dim'),
            latent_dim=conf.get('encoder', 'latent_dim'),
            epsilon_std=1.
        )
        epochs = conf.get('encoder', 'test_epochs')

        X = np.array(wl_data)
        vae.fit(X, X, epochs=epochs)
        vae_dict[wl] = (vae, enc, gen)


    # for wl in samples4lumos:
    for wl in test_wls:
        vae, enc, gen = vae_dict[wl]

        def encode_data(wl_data):
            for i in range(len(wl_data)):
                metrics = wl_data[i].get_metrics()
                encoded_metrics = enc.predict(metrics.reshape(1, -1, 62), batch_size=1)
                wl_data[i].update_metrics(encoded_metrics.reshape(-1), tag='enc')

        encode_data(samples4lumos[wl])
        X, Y = samples4lumos[wl][0].as_vector()
        print(X.shape)

    n_feat = conf.get('encoder', 'latent_dim') + 4 * 2
    lumos_model = LumosModel(n_feat=n_feat)
