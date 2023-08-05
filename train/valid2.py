from train.load_path import load_train_valid
from train.make_dataloader import make_valid_loader
from models.model_run2 import ModelRun2

from tools.io_tools import mkdir_if_missing
from tools.log_tools import setup_logger
from tools.net_tools import set_seed, load_param
from config.parameters import CFG, OUTPUT_DIR

import json
import os
import torch
import torch.nn as nn

#
MODEL_DIR = 'train_model_save_2022-05-25-11-29-51'
which_gpu = 4
which_dataset = 'ubfc2'
if which_dataset in ['vipl']:
    which_v_idx = 4
else:
    which_v_idx = None
which_video_path_prefix = 'video_1x1'
if which_dataset in ['vipl']:
    which_step = 15  # follow previous work
elif which_dataset in ['pure', 'ubfc2', 'pure_ubfc2']:
    # UBFC follow previous work: PulseGAN, DualGAN
    which_step = 30
else:
    raise Exception('Unknown dataset = {}'.format(which_dataset))


def main():
    CFG['MODEL_SAVE_DIR'] = CFG['MODEL_SAVE_DIR'].replace('model_save', 'valid_model_save')
    CFG['DEVICE_ID'] = '{}'.format(which_gpu)

    num_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    set_seed(CFG['SEED'])
    mkdir_if_missing(CFG['MODEL_SAVE_DIR'])
    logger = setup_logger("remote_ppg_RePSS", CFG['MODEL_SAVE_DIR'], if_train=True)
    logger.info("Using {} GPUS".format(num_gpu))
    logger.info("Saving model in the path :{}".format(
        CFG['MODEL_SAVE_DIR']
    ))
    logger.info('MODEL_DIR: {}'.format(
        MODEL_DIR
    ))
    logger.info('which_video_path_prefix: {}'.format(
        which_video_path_prefix
    ))
    logger.info('which_step: {}'.format(
        which_step
    ))
    logger.info(
        'which_v_idx = {}'.format(which_v_idx)
    )
    #
    #
    #
    train_path_list, valid_path_list = load_train_valid(db_name=which_dataset, v_idx=which_v_idx)

    valid_loader = make_valid_loader(valid_path_list, batch_size=CFG['VALID_BATCH_SIZE'], step=which_step,
                                     video_path_prefix=which_video_path_prefix, db_name=which_dataset,
                                     )
    logger.info('-' * 80)
    logger.info('valid_path_list.length = {}'.format(len(valid_path_list)))
    # for x in valid_loader:
    #     pass
    # exit(0)
    print(json.dumps(CFG['MODEL_SAVE_DIR'], indent=4))
    logger.info("CFG:\n{}".format(
        json.dumps(CFG, indent=4)
    ))
    os.environ['CUDA_VISIBLE_DEVICES'] = CFG['DEVICE_ID']
    device = torch.device('cuda')
    #
    #
    #
    model = ModelRun2(inplanes=32, db_name=which_dataset, )
    model.eval()
    logger.info('Model(init) completed!')

    model = model.to(device)
    if device:
        if torch.cuda.device_count() > 1:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
    logger.info('Model(move to GPU) completed!')

    #
    #
    #
    for epoch in range(CFG['EPOCH_NUM']):
        logger.info('-' * 40 + 'epoch_' + str(epoch) + '-' * 40)
        #
        # eval
        #
        model_path = os.path.join(
            OUTPUT_DIR,
            MODEL_DIR,
            'model_{}.pth'.format(epoch),
        )
        if not os.path.isfile(model_path):
            break
        load_param(model, model_path)
        model.forward_valid(
            valid_loader=valid_loader,
            logger=logger,
            device=device,
            epoch=epoch,
            CFG=CFG,
        )


if __name__ == '__main__':
    main()
