import os
import sys
import json
import dill
import copy
import pickle
import numpy as np

from enum import Enum, unique
from collections import defaultdict

from utils import *
from conf import LumosConf
from stat_encoder.fa import FAEncoder
from stat_encoder.pca import PCAEncoder


class RecordEntry(object):

    def __init__(self, inst_type, metrics, jct, ts):
        # raw features
        self.inst_type = inst_type
        self.metrics = metrics
        self.ts = ts
        # raw label
        self.jct = jct
        # rank label
        self.rank = -1


    def feat_as_vector(self):
        pass


    def as_vector(self):
        pass


class DataLoaderOrdinal(object):

    def __init__(self, dump_pth=None):
        self.conf = LumosConf()
        self.ds_root_pth = self.conf.get('dataset', 'path')
        self.vendor_cnt = self.conf.get('dataset', 'vendor_cnt')
        self.__data = None
        self.dump_pth = dump_pth


    def load_data(self):
        if self.dump_pth:
            self.load_data_from_file()
            return
        
        self.__data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

        def is_vendor(v):
            return '.' not in v

        for vendor in os.listdir(self.ds_root_pth):
            if not is_vendor(vendor): continue
            v_pth = os.path.join(self.ds_root_pth, vendor)
            for inst_type in os.listdir(v_pth):
                i_pth = os.path.join(v_pth, inst_type)
                for w in os.listdir(i_pth):
                    [scale, rnd] = w.strip().split('_')[-2:]
                    if rnd not in ['1', '2', '3']: continue
                    workload = '_'.join(w.strip().split('_')[:2])
                    w_pth = os.path.join(i_pth, w)
                    repo_pth = os.path.join(w_pth, 'report.json')
                    metr_pth = os.path.join(w_pth, 'sar.csv')
                    [ts, jct] = mget_json_values(repo_pth, 'timestamp', 'elapsed_time')
                    ts = encode_timestamp(ts)
                    jct = float(jct)
                    header, metrics = read_csv(metr_pth)
                    if not header or not metrics: continue
                    norm_metrics = normalize_metrics(metrics, centralize=True)
                    self.__data[rnd][workload][scale].append(
                        RecordEntry(inst_type, norm_metrics, jct, ts)
                    )


    def get_data(self):
        return self.__data


    def get_data_rankize(self):
        '''
        sort the performance of a workload running with a concrete input scale
        '''
        assert self.__data, 'data not loaded'
        rankize_data = copy.deepcopy(self.__data)
        for rnd, rnd_data in rankize_data.items():
            for wl, wl_data in rnd_data.items():
                for scale in wl_data:
                    scale_data = wl_data[scale]
                    sorted_scale_data = sorted(scale_data, key=lambda x: x.jct)
                    for record in sorted_scale_data:
                        record.rank = sorted_scale_data.index(record)
                    wl_data[scale] = sorted_scale_data

        return rankize_data


    def get_training_data(self, t_inst_type):
        '''
        get the training data that profiled on a concrete instance type
        param:
        @t_inst_type: the instance type that is used for profiling
        '''
        rankize_data = self.get_data_rankize()
        train_data = defaultdict(lambda: defaultdict(lambda: None))
        for rnd, rnd_data in rankize_data.items():
            for wl, wl_data in rnd_data.items():
                profile_scale = 'small'
                predict_scales = ['large', 'huge']
                # TODO: organize the dataset after encoding
                # TODO: encoding: simple statistic-based encoding such as FA, PCA, etc.


    def load_data_from_file(self):
        with open(self.dump_pth, 'rb') as fd:
            self.__data = dill.load(fd)


if __name__ == "__main__":
    conf = LumosConf()
    dump_pth = conf.get('dataset', 'dump_pth_ordinal')
    dataloader = DataLoaderOrdinal()
    # dataloader = DataLoaderOrdinal(dump_pth=dump_pth)
    dataloader.load_data()
    data = dataloader.get_data()
    with open(dump_pth, 'wb') as fd:
        dill.dump(data, fd)
    print(len(data['1']))
    print(len(data['2']))
    print(len(data['3']))
