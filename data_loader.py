import os
import sys
import json
import dill
import pickle
import numpy as np

from enum import Enum, unique
from collections import defaultdict

from utils import *
from conf import LumosConf


# @unique
# class MetricsTag(Enum):
#     '''
#     record the status of the metrics
#     '''
#     RAW = 0 # raw data, unmodified
#     PAD = 1 # after padding
#     ENC = 2 # after being encoded by the VAE


class RecordEntry(object):

    '''
    A record entry of running a workload on a concrete configuration
    '''
    def __init__(self, inst_type, scale, metrics, jct, ts):
        '''
        params:
        @scale: size of input
        @metrics: low-level system metrics data
        @jct: job completion time (encoded)
        @ts: timestamp
        '''
        self.inst_type = inst_type
        self.scale = scale
        self.metrics = metrics
        self.jct = jct
        self.ts = ts
        self.tag = 'raw'


    def get_metrics(self):
        return self.metrics


    def update_metrics(self, new_metrics, tag=None):
        '''
        sometimes metrics need to be updated, e.g., padding or encoding
        params:
        @new_metrics: the metrics data that is about to be updated
        @tag: pad or enc
        '''
        assert tag in ('pad', 'enc'), 'invalid update tag: %s' % tag
        # if tag == 'pad': self.tag = MetricsTag.PAD
        # elif tag == 'enc': self.tag = MetricsTag.ENC
        self.tag = tag

        self.metrics = new_metrics
    

    def as_vector(self):
        '''
        turn this record to a vector that can be fed into a prediction model
        '''
        # assert self.tag == MetricsTag.ENC, 'metrics un-encoded, unable to vectorize'
        assert self.tag == 'enc', 'metrics un-encoded, unable to vectorize'
        conf = LumosConf()
        inst_id = conf.get_inst_id(self.inst_type)
        scale_id = conf.get_scale_id(self.scale)
        X = np.array([inst_id, scale_id, self.ts[0], self.ts[1]])
        X = np.concatenate((X, self.metrics), axis=0)
        Y = self.jct
        return X, Y


class DataLoader(object):
    '''
    Load training or testing data
    '''
    def __init__(self, dump_pth=None):
        self.conf = LumosConf()
        self.ds_root_pth = self.conf.get('dataset', 'path')
        self.vendor_cnt = self.conf.get('dataset', 'vendor_cnt')
        self.__data = defaultdict(lambda: defaultdict(lambda: []))
        self.dump_pth = dump_pth


    def load_data(self):
        if self.dump_pth:
            self.load_data_from_file()
            return
        
        def is_vendor(v):
            return '.' not in v

        for vendor in os.listdir(self.ds_root_pth):
            if not is_vendor(vendor): continue
            pth1 = os.path.join(self.ds_root_pth, vendor)
            for inst_type in os.listdir(pth1):
                pth2 = os.path.join(pth1, inst_type)
                for w in os.listdir(pth2):
                    [scale, _] = w.strip().split('_')[-2:]
                    workload = '_'.join(w.strip().split('_')[:2])
                    pth3 = os.path.join(pth2, w)
                    pth_report = os.path.join(pth3, 'report.json')
                    pth_metrics = os.path.join(pth3, 'sar.csv')
                    [ts, jct] = mget_json_values(pth_report, 'timestamp', 'elapsed_time')
                    ts = encode_timestamp(ts)
                    jct = float(jct)
                    header, metrics = read_csv(pth_metrics)
                    if not header or not metrics: continue
                    norm_metrics = normalize_metrics(metrics)
                    self.__data[workload][vendor].append(
                        RecordEntry(inst_type, scale, norm_metrics, jct, ts))
    
    
    def load_data_from_file(self):
        with open(self.dump_pth, 'rb') as fd:
            self.__data = dill.load(fd)


    def get_data(self):
        return self.__data


if __name__ == "__main__":
    conf = LumosConf()
    dump_pth = conf.get('dataset', 'dump_pth')
    #data_loader = DataLoader(dump_pth=dump_pth)
    data_loader = DataLoader()
    data_loader.load_data()
    data = data_loader.get_data()
    print(len(data))
    print(data.keys())
    print(len(data['hadoop_aggregation']['alibaba']))
    print(len(data['hadoop_aggregation']['huawei']))
    print(len(data['hadoop_aggregation']['tencent']))
    # print(len(data['hadoop_aggregation']['ucloud']))
    with open(dump_pth, 'wb') as fd:
       dill.dump(data, fd)
