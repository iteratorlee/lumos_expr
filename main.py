import os
import sys
import json
import time
import pickle
import random
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
    dump_pth = None
    if conf.get('dataset', 'no_dup'):
        dump_pth = conf.get('dataset', 'dump_pth_no_dup')
    else:
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


    ####### begin test #########

    # for wl in samples4lumos.keys():
    #     wl_data = samples4lumos[wl]
    #     jcts = [d.jct for d in wl_data]
    #     with open('tmp/%s.jcts.dat' % wl, 'wb') as fd:
    #         pickle.dump(jcts, fd)
    # exit(-2)

    ####### end test ###########


    # for wl, in sample_metrics4enc:
    for wl in test_wls:
        wl_data = sample_metrics4enc[wl]
        ### test ###
        jcts = []
        for data in samples4lumos[wl]:
            jcts.append(data.jct)
        with open('res/jcts.dat', 'wb') as fd:
            pickle.dump(jcts, fd)
        exit(-1)
        ############
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


    n_static_feat = 7
    # n_feat = conf.get('encoder', 'latent_dim') + 4 * 2
    # n_feat_naive = 4 * 2
    n_feat_naive = n_static_feat * 2
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
        
        # train lumos model for this workload
        lumos_model = LumosModel(n_feat=n_feat_naive)

        def bin_func(x1, x2):
            return 0 if x1 >= x2 else 1

        def log_func(x1, x2):
            return np.log(x1 / x2)

        def ord_func(x1, x2):
            return 0

        def gen_X_Y(wl_data):
            X, Y = [], []
            cnt = 0
            for i in range(len(wl_data) - 1):
                for j in range(i + 1, len(wl_data)):
                    record1, record2 = wl_data[i], wl_data[j]
                    cnt += 1
                    # print('generating train/valid data, %d/%d' % (cnt, len(wl_data) ** 2), end='\r')
                    x1, jct_1 = record1.as_vector()
                    x2, jct_2 = record2.as_vector()
                    x1_naive = x1[:n_static_feat]
                    x2_naive = x2[:n_static_feat]
                    y = log_func(jct_2, jct_1)
                    # X.append(np.concatenate((x1, x2[:4]), axis=0))
                    X.append(np.concatenate((x1_naive, x2_naive), axis=0))
                    Y.append(y)
            # print()
            return np.asarray(X), np.asarray(Y)
        
        def train_valid_split(length):
            train_ids = random.sample(list(range(length)), int(0.8 * length))
            valid_ids = list(set(range(length)) - set(train_ids))
            return train_ids, valid_ids
        
        print('generating train/valid data for %s' % wl)
        X, Y = gen_X_Y(samples4lumos[wl])
        train_ids, valid_ids = train_valid_split(len(X))
        train_X, train_Y = X[train_ids], Y[train_ids]
        valid_X, valid_Y = X[valid_ids], Y[valid_ids]
        lumos_model.train(train_X, train_Y, valid_X, valid_Y,
            batch_size=conf.get('lumos_model', 'batch_size'),
            epochs=conf.get('lumos_model', 'test_epochs')
            )

        def gen_test_X_Y(wl_data, begin):
            X, Y = [], []
            for record1 in wl_data[begin:]:
                tmp_X, tmp_Y = [], []
                for record2 in wl_data:
                    x1, jct_1 = record1.as_vector()
                    x2, jct_2 = record2.as_vector()
                    x1_naive = x1[:n_static_feat]
                    x2_naive = x2[:n_static_feat]
                    y = func(jct_2 / jct_1)
                    tmp_X.append(np.concatenate((x1_naive, x2_naive), axis=0))
                    tmp_Y.append(y)
                X.append(tmp_X)
                Y.append(tmp_Y)
            return np.asarray(X), np.asarray(Y)

        test_X, test_Y = gen_test_X_Y(samples4lumos[wl], -29)
        test_Y_bar = []
        for tmp_X in test_X:
            tmp_Y_bar = lumos_model.predict(tmp_X)
            test_Y_bar.append(tmp_Y_bar)
        with open('res/test_Y.dat', 'wb') as fd:
            pickle.dump([test_Y_bar, test_Y], fd)
        
        valid_Y_bar = lumos_model.predict(valid_X)
        with open('res/valid_Y.dat', 'wb') as fd:
            pickle.dump([valid_Y_bar, valid_Y], fd)
