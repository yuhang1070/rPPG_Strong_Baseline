import json
import os
import torch
import torch.nn as nn

from losses.cross_snr_loss import CrossSNRLoss
from train.load_path import load_train_valid
from train.make_dataloader import make_train_loader
from models.model_run2 import ModelRun2
from tools.io_tools import mkdir_if_missing
from tools.log_tools import setup_logger
from tools.net_tools import set_seed, load_param
from config.parameters import CFG, OUTPUT_DIR


which_gpu = 5
which_dataset = 'ubfc2'
if which_dataset in ['vipl']:
    which_v_idx = 4
else:
    which_v_idx = None
which_prefix = 'video_1x1'
which_step = 10
which_aug = 'pc'  # patch crop
use_pre_train_bvp_ae = True


def main():
    CFG['MODEL_SAVE_DIR'] = CFG['MODEL_SAVE_DIR'].replace('model_save', 'train_model_save')
    CFG['DEVICE_ID'] = '{}'.format(which_gpu)
    #
    # log param
    #
    num_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    set_seed(CFG['SEED'])
    mkdir_if_missing(CFG['MODEL_SAVE_DIR'])
    logger = setup_logger("RePSS", CFG['MODEL_SAVE_DIR'], if_train=True)
    logger.info("Using {} GPUS".format(num_gpu))
    logger.info("Saving model in the path :{}".format(
        CFG['MODEL_SAVE_DIR']
    ))
    print(json.dumps(CFG['MODEL_SAVE_DIR'], indent=4))
    logger.info("CFG:\n{}".format(
        json.dumps(CFG, indent=4)
    ))
    logger.info(
        'which_dataset  = {}'.format(which_dataset)
    )
    logger.info(
        'which_step  = {}'.format(which_step)
    )
    logger.info(
        'which_prefix  = {}'.format(which_prefix)
    )
    logger.info(
        'which_v_idx = {}'.format(which_v_idx)
    )
    logger.info(
        'which_gpu = {}'.format(which_gpu)
    )
    logger.info(
        'which_aug  = {}'.format(which_aug)
    )
    logger.info(
        'use_pre_train_bvp_ae = {}'.format(use_pre_train_bvp_ae)
    )
    #
    # get data-loader
    #
    train_path_list, valid_path_list = load_train_valid(db_name=which_dataset, v_idx=which_v_idx)
    train_loader = make_train_loader(
        train_path_list,
        batch_size=CFG['TRAIN_BATCH_SIZE'],
        step=which_step,
        video_path_prefix=which_prefix,
        which_aug=which_aug,
        db_name=which_dataset,
    )
    logger.info('-' * 80)
    logger.info('train_path_list.length = {}'.format(len(train_path_list)))
    #
    # create model
    #
    os.environ['CUDA_VISIBLE_DEVICES'] = CFG['DEVICE_ID']
    device = torch.device('cuda')
    model = ModelRun2(
        inplanes=32,
        db_name=which_dataset,
    )
    model.init_random()
    logger.info('Model(init) completed!')
    #
    # model to device
    #
    model = model.to(device)
    if device:
        if torch.cuda.device_count() > 1:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
    #
    # load pre-train BVP AE
    #
    if use_pre_train_bvp_ae:
        BVP_AE_DIR = os.path.join(OUTPUT_DIR, 'model_save_2021-10-26-16-33-04')
        assert os.path.isdir(BVP_AE_DIR)
        BVP_AE_PATH = os.path.join(BVP_AE_DIR, 'model_6.pth')
        assert os.path.isfile(BVP_AE_PATH)
        load_param(model.bvp_ae, BVP_AE_PATH)
        #
        # frozen BVP AutoEncoder
        #
        model.bvp_ae.frozen()
    else:
        #
        # frozen BVP Encoder
        #
        model.bvp_enc.frozen()
    logger.info('Model(move to GPU) completed!')
    #
    #
    #
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG['lr'],
    )
    logger.info('Optimizer completed!')
    loss_psd_func = CrossSNRLoss(
        clip_length=256,
        device=device,
        batch_size=CFG['TRAIN_BATCH_SIZE'],
    )
    #
    #
    #
    save_period = CFG['SAVE_PERIOD']
    for epoch in range(CFG['EPOCH_NUM']):
        logger.info('-' * 40 + 'epoch_' + str(epoch) + '-' * 40)
        model.forward_train(
            train_loader=train_loader,
            optimizer=optimizer,
            logger=logger,
            device=device,
            epoch=epoch,
            CFG=CFG,
            loss_psd_func=loss_psd_func,
        )
        #
        # save
        #
        if epoch % save_period == 0:
            torch.save(
                model.state_dict(),
                os.path.join(CFG['MODEL_SAVE_DIR'], 'model_{}.pth'.format(epoch)),
            )


if __name__ == '__main__':
    main()
