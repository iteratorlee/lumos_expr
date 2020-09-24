import os
import json
from utils import singleton


@singleton
class LumosConf(object):

    def __init__(self):
        with open('conf/lumos.json') as fd:
            self.__conf = json.load(fd)
        with open('conf/inst_conf.json') as fd:
            self.__inst_conf = json.load(fd)


    def get(self, *key):
        tmp = self.__conf[key[0]]
        if len(key) > 1:
            for i in range(len(key) - 1):
                tmp = tmp[key[i + 1]]
        return tmp


    def get_inst_id(self, inst):
        return self.__inst_conf[inst]


    def get_scale_id(self, scale):
        scale_arr = ('tiny', 'small', 'large', 'huge')
        assert scale in scale_arr, 'invalid scale: %s' % scale
        return scale_arr.index(scale) / 4
