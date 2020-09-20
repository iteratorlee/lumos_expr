import os
import json
from utils import singleton


@singleton
class LumosConf(object):

    def __init__(self):
        with open('conf/lumos.json') as fd:
            self.__conf = json.load(fd)


    def get(self, *key):
        tmp = self.__conf[key[0]]
        if len(key) > 1:
            for i in range(len(key) - 1):
                tmp = tmp[key[i + 1]]
        return tmp
