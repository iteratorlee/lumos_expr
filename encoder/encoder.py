import os
import sys
import json
import time

import tensorflow as tf


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
        load_encoders from TF dump files
        '''
        pass

        
    def encode_metrics(self, metrics):
        return metrics
