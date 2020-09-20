import os
import sys
import json
import numpy as np

from conf import LumosConf


class DataLoader(object):

    def __init__(self):
        conf = LumosConf()
        ds_root_pth = conf.get('dataset', 'path')


    def load_data(self):
        pass


    def get_data(self):
        pass


if __name__ == "__main__":
    data_loader = DataLoader()
