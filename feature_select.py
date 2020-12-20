import json
import random
import numpy as np
from conf import LumosConf
from data_loader_ordinal import DataLoaderOrdinal


def cal_std_and_cof(metrics):
    return np.std(metrics, axis=0), np.corrcoef(metrics.T)


def ana_metrics(metrics_data):
    stat_res = []
    for metrics in metrics_data:
        std, cof = cal_std_and_cof(metrics)
        stat_res.append((std, cof))
    return stat_res


def select_features(metrics_data):
    return []


if __name__ == "__main__":
    conf = LumosConf()
    dump_pth = conf.get('dataset', 'dump_pth_ordinal_with_truc_v1')
    dataloader = DataLoaderOrdinal(dump_pth=dump_pth)
    dataloader.load_data_by_interval(interval=1)
    data = dataloader.get_data()
    to_select_scale = 'large'
    metrics_data = []
    for wl, wl_data in data['1'].items():
        scale_data = wl_data[to_select_scale]
        metrics_data.append(random.sample(scale_data, 1)[0].metrics)
    # feature_idxes = select_features(metrics_data)
    ana_metrics(metrics_data)
