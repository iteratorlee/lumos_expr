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
from stat_encoder.fft_stat import FFTStatEncoder


class RecordEntry(object):

    def __init__(self, inst_type, metrics, raw_metrics, jct, ts):
        # raw features
        self.inst_type = inst_type
        self.metrics = metrics
        self.raw_metrics = raw_metrics
        self.ts = ts
        # raw label
        self.jct = jct
        # rank label
        self.rank = -1


    def __repr__(self):
        repr_dict = {
            'inst_type': self.inst_type,
            'metrics.shape': self.metrics.shape,
            'ts': self.ts,
            'jct': self.jct,
            'rank': self.rank
        }
        import json
        return json.dumps(repr_dict, indent=4)


class DataLoaderOrdinal(object):

    def __init__(self, dump_pth=None):
        self.conf = LumosConf()
        self.ds_root_pth = self.conf.get('dataset', 'path')
        self.vendor_cnt = self.conf.get('dataset', 'vendor_cnt')
        self.__data = None
        self.dump_pth = dump_pth
        # sampling interval
        self.sampling_interval = 5


    def load_data(self):
        '''
        old load_data, interval=5s
        '''
        self.sampling_interval = 5
        if self.dump_pth:
            self.__load_data_from_file()
            return

        self.__data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

        def is_vendor(v):
            return '.' not in v

        for vendor in os.listdir(self.ds_root_pth):
            if not is_vendor(vendor): continue
            v_pth = os.path.join(self.ds_root_pth, vendor, '5 second')
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
                    raw_metrics = np.array(metrics)
                    self.__data[rnd][workload][scale].append(
                        RecordEntry(inst_type, norm_metrics, raw_metrics, jct, ts)
                    )


    def load_data_by_interval(self, interval=5):
        '''
        load data with specific sampling interval
        '''
        assert interval in (1, 5), 'invalid interval'
        self.sampling_interval = interval
        if interval == 5:
            self.load_data()
            return

        if self.dump_pth:
            self.__load_data_from_file()
            return

        self.__data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

        def is_vendor(v):
            return '.' not in v

        for vendor in os.listdir(self.ds_root_pth):
            if not is_vendor(vendor): continue
            v_pth_1 = os.path.join(self.ds_root_pth, vendor, '1 second')
            for inst_type in os.listdir(v_pth_1):
                i_pth = os.path.join(v_pth_1, inst_type)
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
                    raw_metrics = np.array(metrics)
                    self.__data[rnd][workload][scale].append(
                        RecordEntry(inst_type, norm_metrics, raw_metrics, jct, ts)
                    )
            v_pth_2 = os.path.join(self.ds_root_pth, vendor, '5 second')
            for inst_type in os.listdir(v_pth_2):
                i_pth = os.path.join(v_pth_2, inst_type)
                for w in os.listdir(i_pth):
                    [scale, rnd] = w.strip().split('_')[-2:]
                    if scale in ('tiny', 'small'): continue
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
                    raw_metrics = np.array(metrics)
                    self.__data[rnd][workload][scale].append(
                        RecordEntry(inst_type, norm_metrics, raw_metrics, jct, ts)
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


    def get_train_test_data(self, train_scale='tiny', test_wl=''):
        '''
        get the training data that profiled on a concrete instance type
        param:
        @t_inst_type: the instance type that is used for profiling
        @test_wl: the workload that is to be used for testing
        '''
        rankize_data = self.get_data_rankize()
        assert test_wl in self.__data['1'], 'invalid test workload'
        fft_stat_encoder = FFTStatEncoder()
        conf = LumosConf()

        train_data = defaultdict(lambda: defaultdict(lambda: {
            'X': [],
            'Y': []
        }))
        test_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            'X': [],
            'Y': []
        })))

        predict_scales = ['tiny', 'small', 'large', 'huge']
        if train_scale == 'small':
            predict_scales.remove('tiny')

        for rnd, rnd_data in rankize_data.items():
            for wl, wl_data in rnd_data.items():
                if wl == test_wl: continue
                for record1 in wl_data[train_scale]:
                    t_inst_type = record1.inst_type
                    test_conf = conf.get_inst_detailed_conf(t_inst_type, format='list')
                    test_metrics_vec = fft_stat_encoder.encode(record1.metrics, record1.raw_metrics, sampling_interval=self.sampling_interval)
                    for scale in predict_scales:
                        target_scale = conf.get_scale_id(scale)
                        for record2 in wl_data[scale]:
                            target_conf = conf.get_inst_detailed_conf(record2.inst_type, format='list')
                            target_rank = record2.rank
                            X = test_conf.copy()
                            X.extend(target_conf)
                            X.append(target_scale)
                            X.extend(test_metrics_vec)
                            train_data[rnd][t_inst_type]['X'].append(X)
                            train_data[rnd][t_inst_type]['Y'].append(target_rank)

        for rnd, rnd_data in rankize_data.items():
            wl_data = rnd_data[test_wl]
            for record1 in wl_data[train_scale]:
                t_inst_type = record1.inst_type
                test_conf = conf.get_inst_detailed_conf(t_inst_type, format='list')
                test_metrics_vec = fft_stat_encoder.encode(record1.metrics, record1.raw_metrics, sampling_interval=self.sampling_interval)
                for scale in predict_scales:
                    target_scale = conf.get_scale_id(scale)
                    for record2 in wl_data[scale]:
                        target_conf = conf.get_inst_detailed_conf(record2.inst_type, format='list')
                        target_rank = record2.rank
                        X = test_conf.copy()
                        X.extend(target_conf)
                        X.append(target_scale)
                        X.extend(test_metrics_vec)
                        test_data[rnd][t_inst_type][scale]['X'].append(X)
                        test_data[rnd][t_inst_type][scale]['Y'].append(target_rank)

        return train_data, test_data


    def __load_data_from_file(self):
        with open(self.dump_pth, 'rb') as fd:
            self.__data = dill.load(fd)


if __name__ == "__main__":
    conf = LumosConf()
    dump_pth = conf.get('dataset', 'dump_pth_ordinal_1s')
    # dataloader = DataLoaderOrdinal()
    dataloader = DataLoaderOrdinal(dump_pth=dump_pth)
    dataloader.load_data_by_interval(interval=1)
    data = dataloader.get_data()
    # with open(dump_pth, 'wb') as fd:
        # dill.dump(data, fd)
    print(len(data['1']))
    print(len(data['2']))
    print(len(data['3']))
    train_data, test_data = dataloader.get_train_test_data(test_wl='spark_pagerank')
