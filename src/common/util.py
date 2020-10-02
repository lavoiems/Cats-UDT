import os
import json
import numpy as np
import torch
import torchvision.utils as vutils
import math


def set_paths(args):
    args.save_path = os.path.join(args.root_path, args.model, args.run_name)
    args.model_path = os.path.join(args.save_path, 'model')
    args.log_path = os.path.join(args.save_path, 'log')


def create_paths(save_path, model_path, log_path):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)


def dump_args(args):
    if args.reload and os.path.isfile(os.path.join(args.save_path, 'args.json')):
        return
    args_dict = get_args_dict(args)
    json.dump(args_dict, open(os.path.join(args.save_path, 'args.json'), 'w'))


def get_args(save_path):
    args_path = os.path.join(save_path, 'args.json')
    with open(args_path, 'r') as f:
        return json.load(f)


def get_args_dict(args):
    builtin = ('basestring', 'bool', 'complex', 'dict', 'float', 'int',
               'list', 'long', 'str', 'tuple')
    args_dict = {k: v for k, v in vars(args).items()
                 if type(v).__name__ in builtin}
    return args_dict


def normalize(x):
    x.clamp_(-1, 1)
    x.add_(1).mul_(0.5)
    return x


def save_image(x, filename):
    ncol = int(math.sqrt(len(x)))
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


def normalize_channels(image, concat=np.concatenate):
    if image.shape[1] == 1:
        return concat([image, image, image], 1)
    return image


def compare(base, transfer):
    left = base.copy()
    right = transfer.copy()
    left = normalize_channels(left)
    right = normalize_channels(right)
    return np.concatenate([left, right], axis=3)


def save_models(models, iteration, model_path, checkpoint):
    for name, model in models.items():
        path = os.path.join(model_path, f'{name}:{iteration}')
        torch.save(model.state_dict(), path)
        rmpath = os.path.join(model_path, f'{name}:{iteration-checkpoint}')
        if os.path.exists(rmpath):
            os.remove(rmpath)


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes, device=labels.device)
    return y[labels]


def sample(iterator, loader, expected_size=None):
    try:
        batch = next(iterator)
        if expected_size and batch.shape[0] != expected_size:
            batch = next(iterator)
        return batch, iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator
