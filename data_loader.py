import os
import sys
import json
import numpy as np

from collections import defaultdict

from conf import LumosConf
from utils import read_csv, get_json_value, mget_json_values, encode_timestamp


class RecordEntry(object):
    '''
    A record entry of running a workload on a concrete configuration
    '''
    def __init__(self, inst_type, scale, metrics, jct, ts):
        '''
        params:
        @scale: size of input
        @metrics: low-level system metrics data
        @jct: job completion time
        @ts: timestamp
        '''
        self.inst_type = inst_type
        self.scale = scale
        self.metrics = metrics
        self.jct = jct
        self.ts = ts
    

    def as_vector(self):
        '''
        turn this record to a vector that can be fed into a prediction model
        '''
        return []


class DataLoader(object):
    '''
    Load training or testing data
    '''
    def __init__(self):
        self.conf = LumosConf()
        self.ds_root_pth = self.conf.get('dataset', 'path')
        self.vendor_cnt = self.conf.get('dataset', 'vendor_cnt')
        self.__data = defaultdict(lambda: defaultdict(lambda: []))

    def load_data(self):
        
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
                    self.__data[workload][vendor].append(
                        RecordEntry(inst_type, scale, metrics, jct, ts))


    def get_data(self):
        return self.__data


if __name__ == "__main__":
    data_loader = DataLoader()
    data_loader.load_data()
    data = data_loader.get_data()
    print(len(data))
    print(len(data['hadoop_aggregation']))
    print(len(data['hadoop_aggregation']['alibaba']))
    print(len(data['hadoop_aggregation']['huawei']))
    print(len(data['hadoop_aggregation']['tencent']))
    print(len(data['hadoop_aggregation']['ucloud']))