import logging
import os
import sys
import time


def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(
                os.path.join(
                    save_dir,
                    "train_log_{}.txt".format(time.strftime('%Y-%m-%d-%H-%M-%S'))
                ), mode='w'
            )
        else:
            fh = logging.FileHandler(
                os.path.join(
                    save_dir,
                    "test_log_{}.txt".format(time.strftime('%Y-%m-%d-%H-%M-%S'))
                ),
                mode='w'
            )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class CmdLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
