import json

def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]

    return inner


def read_csv(csv_file_path):
    with open(csv_file_path, 'r') as fd:
        lines = fd.readlines()
        header, content = lines[0], lines[1:]
        content = [line.strip().split(',') for line in content]
        for i in range(len(content)):
            for j in range(len(content[i]) - 1):
                ele = content[i][j + 1]
                if ele == 'lo': # IFACE
                    content[i][j + 1] = 0
                elif ele.replace('.', '').isdigit():
                    content[i][j + 1] = float(ele)
                else:
                    return None, None
        return header, content


def get_json_value(json_file_path, *keys):
    with open(json_file_path, 'r') as fd:
        data = json.load(fd)
        tmp = data[keys[0]]
        if len(keys) > 1:
            for i in range(len(keys) - 1):
                tmp = tmp[keys[i + 1]]
        return tmp


def mget_json_values(json_file_path, *key_arr):
    with open(json_file_path, 'r') as fd:
        data = json.load(fd)
        ret = []
        for keys in key_arr:
            if isinstance(keys, list):
                tmp = data[keys[0]]
                if len(keys) > 1:
                    for i in range(len(keys) - 1):
                        tmp = tmp[keys[i + 1]]
                    ret.append(tmp)
            elif isinstance(keys, str):
                ret.append(data[keys])
        return ret


def encode_timestamp(ts):
    return 0