import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.benchmark = False


def load_param(model, model_path: str):
    param_dict = torch.load(model_path)
    for i in param_dict:
        # print(i)
        model.state_dict()[i].copy_(param_dict[i])
    return model


def remove_prefix(state_dict, prefix):
    """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
    print('remove prefix \'{}\''.format(prefix))
    func = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {func(key): value for key, value in state_dict.items()}


def check_keys(model, pretrained_state_dict):
    # Missing keys:0
    # Unused checkpoint keys:0
    # Used keys:300
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True
