import os
import sys
import json
import time
import numpy as np

from collections import defaultdict
from keras.models import load_model

from utils import *
from conf import LumosConf
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
    
    for wl in samples4enc:
        padding_data(samples4enc[wl], max_len=max_lens[wl])

    # for wl in samples4lumos:
    #     padding_data(samples4lumos[wl], max_len=max_lens[wl])
        
    # samples4enc is used to train the encoder model
    # samples4lumos is used to train the prediction model

    sample_metrics4enc = defaultdict(lambda: [])
    for wl, _data in samples4enc.items():
        sample_metrics4enc[wl].extend([ele.get_metrics() for ele in _data])
    
    # sample_metrics4lumos = defaultdict(lambda: [])
    # for wl, _data in samples4lumos.items():
        # sample_metrics4lumos[wl].extend([ele.get_metrics() for ele in _data])

    for wl, _data in sample_metrics4enc.items():    
        vae, enc, gen = create_lstm_vae(
            input_dim=_data[0].shape[1],
            timesteps=_data[0].shape[0],
            batch_size=conf.get('encoder', 'batch_size'),
            intermediate_dim=conf.get('encoder', 'intermediate_dim'),
            latent_dim=conf.get('encoder', 'latent_dim'),
            epsilon_std=1.
        )
        epochs = conf.get('encoder', 'epochs')

        X = np.array(_data)
        vae.fit(X, X, epochs=epochs)

        vae.save(os.path.join(conf.get('encoder', 'dump_pth'), '%s.ep_%d.enc.h5' % (wl, epochs)))
