import os
import sys
import json
import time
import keras
import numpy as np

from keras.regularizers import l1
from keras.layers import Dense, Input
from keras.models import Sequential, load_model

from utils import *
from conf import LumosConf
from data_loader import DataLoader
from third_party.keras_lstm_vae.lstm_vae import create_lstm_vae

class LumosModel(object):

    def __init__(self, n_feat, layers=3, lr=0.001, l1_scalar=0.1, \
        loss='mean_squared_error', optimizer_name='sgd', final_active='linear'):
        conf = LumosConf()
        model_name_prefix = conf.get('lumos_model', 'model_name_prefix')
        self.model_name = '%s_%d_%f_%f' % (model_name_prefix, layers, lr, l1_scalar)
        self.model_pth = conf.get('lumos_model', 'model_pth')
        neurons_per_layer = conf.get('lumos_model', 'neurons_per_layer')
        self.model = Sequential()
        self.model.add(Dense(units=neurons_per_layer, input_shape=(n_feat,), \
            kernel_regularizer=l1(l1_scalar), activation='relu'))
        for _ in range(layers - 1):
            self.model.add(Dense(units=neurons_per_layer, activation='relu'))
        self.model.add(Dense(units=1, activation=final_active))

        self.model.summary()
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(lr=lr)
        elif optimizer_name == 'sgd':
            optimizer = keras.optimizers.SGD(lr=lr)
        self.model.compile(loss=loss, optimizer=optimizer)


    def train(self, X, Y, valid_X, valid_Y, batch_size, epochs, shuffle=True):
        timer = Timer()
        print('begin to train model %s' % self.model_name)
        timer.start()
        self.model.fit(
            X,
            Y,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=shuffle,
            validation_data=(valid_X, valid_Y)
        )
        timer.stop()
        print('model %s trained, elapsed_time=%.3f' % \
            (self.model_name, timer.get_elasped_time()))
        
    

    def predict(self, X):
        Y_bar = self.model.predict(X)
        return Y_bar


if __name__ == "__main__":
    model = LumosModel(n_feat=15)
