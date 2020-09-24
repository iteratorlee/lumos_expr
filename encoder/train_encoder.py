import os
import json

import numpy as np

from collections import defaultdict

from conf import LumosConf
from data_loader import DataLoader
from third_party.keras_lstm_vae.lstm_vae import create_lstm_vae


def get_samples(data):
    samples = defaultdict(lambda: [])
    return samples


if __name__ == "__main__":
    conf = LumosConf()
    dump_pth = conf.get('dataset', 'dump_pth')
    data_loader = DataLoader(dump_pth=dump_pth)
    data_loader.load_data()
    data = data_loader.get_data()