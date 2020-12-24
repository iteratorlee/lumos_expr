import os
import sys
import json
import dill
import copy
import argparse
import numpy as np

from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor

from utils import *
from conf import LumosConf
from data_loader_ordinal import DataLoaderOrdinal


def cal_top_3_acc(results):
    return 0


def cal_err(results):
    return 0, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='grid search')
    parser.add_argument('-j', '--n_jobs', help='number of jobs running parallel', type=int, default=None)
    args = parser.parse_args()

    # dataset options
    op_truncate = [True, False]
    op_ordinal = [True, False]
    # model options
    op_max_depth = [3, 4, 5]
    op_n_estimators = [10, 40, 70, 100]
    op_criterion = ['mse', 'mae']
    op_max_features = ['auto', 'sqrt', 'log2', 0.5]
    
    conf = LumosConf()
    dump_pth = conf.get('dataset', 'dump_pth_ordinal_with_truc_v1')
    dataloader = DataLoaderOrdinal(dump_pth=dump_pth)
    dataloader.load_data_by_interval(interval=1)
    data = dataloader.get_data()
    workloads = data['1'].keys()

    best_conf = ''

    # making the model options static, tuning the dataset options
    # making the testing instance type and round static, tuning in the end
    model_init_ops = {
        'op_d': 3,
        'op_e': 10,
        'op_c': 'mse',
        'op_f': 'auto'
    }
    best_op_t, best_op_o = None, None
    best_top_3_acc = .0
    best_abs_err = 0x7ffffff
    best_rel_err = 1.0
    init_rnd, init_t_inst_type = '1', 'g6.large'
    for op_t in op_truncate:
        for op_o in op_ordinal:
            dmp_pre = conf.get('dataset', 'train_test_dump_prefix')
            dmp_suf = 'o%d_t%d' % (op_o, op_t)
            results = defaultdict(lambda: {})
            for wl in workloads:
                wl_pth = os.path.join(dmp_pre, '%s_%s.pkl' % (wl, dmp_suf))
                train_data, test_data = None, None
                with open(wl_pth, 'rb') as fd:
                    (train_data, test_data) = dill.load(fd)
                train_X = train_data[init_rnd][init_t_inst_type]['X']
                train_Y = train_data[init_rnd][init_t_inst_type]['Y']
                regressor = RandomForestRegressor(
                    n_estimators=model_init_ops['op_e'],
                    criterion=model_init_ops['op_c'],
                    max_depth=model_init_ops['op_d'],
                    max_features=model_init_ops['op_f'],
                    n_jobs=args.n_jobs
                )
                regressor.fit(train_X, train_Y)
                for scale, test_XY[init_rnd][init_t_inst_type].items():
                    test_X, test_Y = test_XY['X'], test_XY['Y']
                    test_Y_bar = regressor.predict(test_X)
                    results[wl][scale] = {
                        'test_Y_bar': test_Y_bar,
                        'test_Y': test_Y
                    }
            top_3_acc = cal_top_3_acc(results)
            abs_err, rel_err = cal_err(results) 

            if top_3_acc > best_top_3_acc:
                best_op_o = op_o
                best_op_t = op_t
            elif top_3_acc == best_top_3_acc:
                if abs_err < best_abs_err:
                    best_op_o = op_o
                    best_op_t = op_t
    print(best_op_o, best_op_t)