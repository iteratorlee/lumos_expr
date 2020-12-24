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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='n jobs')
    parser.add_argument('-j', '--n_jobs', help='number of jobs running parallel', type=int, default=None)
    parser.add_argument('-i', '--job_id', help='id the current job', type=int, default=0)
    parser.add_argument('-s', '--start', help='workload start index', type=int, default=0)
    parser.add_argument('-e', '--end', help='workload end index', type=int, default=30)
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
    workloads = sorted(list(data['1'].keys()))

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: \
        defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {}))))))

    for op_t in op_truncate:
        for op_o in op_ordinal:
            dmp_pre = conf.get('dataset', 'train_test_dump_prefix')
            dmp_suf = 'o%d_t%d' % (op_o, op_t)
            for wl in workloads[args.start: args.end]:
                wl_pth = os.path.join(dmp_pre, '%s_%s.pkl' % (wl, dmp_suf))
                train_data, test_data = None, None
                with open(wl_pth, 'rb') as fd:
                    (train_data, test_data) = dill.load(fd)
                for rnd in ['1', '2', '3']:
                    for t_inst_type in train_data[rnd].keys():
                        train_X = train_data[rnd][t_inst_type]['X']
                        train_Y = train_data[rnd][t_inst_type]['Y']
                        for op_d in op_max_depth:
                            for op_e in op_n_estimators:
                                for op_c in op_criterion:
                                    for op_f in op_max_features:
                                            op_code = '%s_%s_%s_t%d_o%d_d%d_e%d_c%s_f%s' \
                                                % (wl, t_inst_type, rnd, op_t, op_o, op_d, op_e, op_c, op_f)
                                            print('processing hpt with op_code=%s' % op_code)
                                            regressor = RandomForestRegressor(
                                                n_estimators=op_e,
                                                criterion=op_c,
                                                max_depth=op_d,
                                                max_features=op_f,
                                                n_jobs=args.n_jobs
                                            )
                                            regressor.fit(train_X, train_Y)
                                            for scale, test_XY in test_data[rnd][t_inst_type].items():
                                                test_X = test_XY['X']
                                                test_Y = test_XY['Y']
                                                test_Y_bar = regressor.predict(test_X)
                                                test_r_bar = np.argsort(test_Y_bar)
                                                results[op_code][rnd][t_inst_type][wl][scale] \
                                                    = {
                                                        'test_Y_bar': test_Y_bar,
                                                        'test_Y': test_Y
                                                    }

    with open(os.path.join(conf.get('dataset', 'hpt_results_pth_prefix'), 'results.%d.pkl' % args.job_id), 'wb') as fd:
        dill.dump(results, fd)
