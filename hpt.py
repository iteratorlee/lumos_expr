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
    # parser = argparse.ArgumentParser(description='Specify hyper-parameters in lumos model')
    
    # # dataset related arguments
    # parser.add_argument('--truncate', '-t', help='find key stage of the time series via truncating', default=True)
    # parser.add_argument('--ordinal', '-o', help='set the label as ordinal ranking or using the raw performance value', default=True)

    # # random forest model related arguments
    # parser.add_argument('--max_depth', '-d', help='max depth of the random forest model', default=3)
    # parser.add_argument('--n_estimators', '-e', help='the number of trees in the forest', default=100)
    # parser.add_argument('--criterion', '-c', help='the function to measure the quality of a split', default='gini')
    # parser.add_argument('--max_features', '-f', help='#feature when looking for a split', default='auto')

    # args = parser.parse_args()

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
