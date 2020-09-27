import os
import sys
import json
import time
import numpy as np

from utils import *
from conf import LumosConf

from keras.regularizers import l1
from keras.layers import Dense, Input
from keras.models import Sequential, load_model

class LumosModel(object):

    def __init__(self, n_feat, layers=3, lr=0.01, l1_scalar=0.1, \
        loss='binary_crossentropy', optimizer='adam'):
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
        self.model.add(Dense(units=1, activation='sigmoid'))

        self.model.summary()
        self.model.compile(loss=loss, optimizer=optimizer)
    

    def train(self, X, Y, batch_size, epochs, shuffle=True):
        timer = Timer()
        print('begin to train model %s' % self.model_name)
        timer.start()
        self.model.fit(
            X,
            Y,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=shuffle
        )
        timer.stop()
        print('model %s trained, elapsed_time=%.3f' % \
            (self.model_name, timer.get_elasped_time()))
        
    

    def predict(self, X):
        Y_bar = self.model.predict(X)
        return Y_bar


if __name__ == "__main__":
    model = LumosModel(n_feat=15)
