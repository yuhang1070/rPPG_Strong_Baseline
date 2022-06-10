import errno
import os
import json
import os.path as osp
import joblib


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print('=> Warning: no file found at "{}" (ignored)'.format(path))
    return isfile


def read_json(filepath):
    with open(filepath, 'r') as f:
        obj = json.load(f)
        f.close()
    return obj


def write_json(obj, filepath):
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))
        f.close()


def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        obj = joblib.load(f)
        f.close()
    return obj


def dump_pkl(obj, filepath):
    with open(filepath, 'wb') as f:
        joblib.dump(obj, filepath)
        f.close()
