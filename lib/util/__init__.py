import logging
import os

import numpy as np
import torch
import yaml

__all__ = [
    'init_logging', 'AverageMeter', 'load_state_dict_safely', 'save_state_dict_safely',
    'save_cmt_and_txt', 'load_config',
]


def init_logging(log_filename):
    """
        Init for logging
    """
    logging.basicConfig(
                    level    = logging.INFO,
                    format   = '%(asctime)s: %(message)s',
                    datefmt  = '%m-%d %H:%M:%S',
                    filename = log_filename,
                    filemode = 'w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M:%S')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_state_dict_safely(module, state_dict):
    '''
        load_state_dict with try except 
    '''
    logging.info('#Start Loading State_dict.')
    model_type = str(type(module))
    if 'parallel' in model_type or 'Parallel' in model_type:
        module = module.module
        logging.info('The initial type is {:s}'.format(model_type))
    logging.info('load state_dict for {:s}'.format(module.__class__.__name__))
    try:
        module.load_state_dict(state_dict)
        logging.info('Successfully loaded.')
    except RuntimeError as e:
        logging.info(e)
        logging.info('Set [strict=False] and try to load state_dict.')
        try:
            module.load_state_dict(state_dict, strict=False)
            logging.info('Successfully loaded.')
        except RuntimeError as e:
            logging.info(e)
            logging.info('#[Warning]: Loading Fail')

    logging.info('#Finish Loading State_dict.')


def save_state_dict_safely(model, save_path, optimizer=None, verbose=False):
    '''
        save state_dict for self-designed model, not nn.module
    '''
    model.eval()
    model_type = str(type(model))
    if 'parallel' in model_type or 'Parallel' in model_type:
        model = model.module

    state_dict = {}
    for key, module in model._modules.items():
        model_type = str(type(module))
        if 'parallel' in model_type or 'Parallel' in model_type:
            module = module.module
        state_dict[key] = module.state_dict()

    state_dict = {'model': state_dict}
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()

    torch.save(state_dict, save_path)
    if verbose:
        logging.info('All Keys in model.state_dict: {:s}'.format(','.join(list(state_dict['model'].keys()))))
        logging.info('Save_Path: {:s}'.format(save_path))


def save_cmt_and_txt(name_list, hyp_list, func_idx2word, save_path=None):
    ctm_list = []
    txt_list = []
    for i in range(len(name_list)):
        hyp = hyp_list[i]
        start_ctm = 0.0
        hyp_word = func_idx2word(hyp)
        txt_list.append('{:s},{:s}\n'.format(name_list[i], ' '.join(hyp_word)))
        for j in range(len(hyp_word)):
            ctm_list.append('{:s} 1 {:.3f} {:.3f} {:s}\n'.format(
                name_list[i], start_ctm, start_ctm+0.001, hyp_word[j]))
            start_ctm = start_ctm+0.001
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        open(save_path+'.cmt', 'w').writelines(ctm_list)
        open(save_path+'.txt', 'w').writelines(sorted(txt_list, key=lambda x: x.split(',')[0]))
    return ctm_list, txt_list


def save_feat(name_list, feature_list, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for (name, feat) in zip(name_list, feature_list):
        np.save(
            os.path.join(save_path, name+'.npy'), feat
        )


def load_config(path: str) -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg
