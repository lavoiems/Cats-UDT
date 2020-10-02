from importlib import import_module
import time
import os
import argparse
import torch
import numpy as np
from common.util import set_paths, create_paths, dump_args
import models


def parse_args(_models_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', default='test', help='Name of the experiment')
    parser.add_argument('--run-id', type=str, default=str(time.time()))
    parser.add_argument('--root-path', default='./experiments/', help='Root path where artefacts are savec')
    parser.add_argument('--server', type=str, default=None, help='Server path if using Visdom')
    parser.add_argument('--port', type=int, default=None, help='Port if using Visdom')
    parser.add_argument('--visdom', action='store_true', help='Flag to use visdom, otherwise use tensorboard')
    parser.add_argument('--reload', action='store_true', help='Flag to reload an experiment at its latest iteration')
    parser.add_argument('--checkpoint', type=int, default=1000, help='Number of iteration per checkpoints')
    parser.add_argument('--iterations', type=int, default=50001, help='Total number of iterations for training')
    parser.add_argument('--evaluate', type=int, default=1000, help='Number of iterations per evaluate')
    parser.add_argument('--train-batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--valid-split', type=float, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--cuda', action='store_true', help='Flag to use cuda')
    subparsers = parser.add_subparsers(dest='model')
    subparsers.required = True
    for name, model in _models_.items():
        if not hasattr(model, 'parse_args') and not hasattr(model, 'execute'):
            continue
        model_parser = subparsers.add_parser(name)
        model.parse_args(model_parser)
        model_parser.set_defaults(func=model.execute)
    return parser.parse_args()


if __name__ == '__main__':
    root = os.path.dirname(os.path.realpath(__file__))
    models_root = os.path.join(root, 'models')
    models_path = [os.path.join(root, f) for f in os.listdir(models_root)]
    models_path = [m for m in models_path if not os.path.isfile(m)]
    models_name = [p.split('/')[-1] for p in models_path]
    _models_ = {name: import_module('.'.join(('models', name))) for name in models_name}
    args = parse_args(_models_)
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    args.run_name = f'{args.exp_name}_{args.run_id}-{args.seed}'
    set_paths(args)
    create_paths(args.save_path, args.model_path, args.log_path)
    dump_args(args)

    if args.visdom:
        from common.visualise import Visualiser
        args.visualiser = Visualiser(args.server, args.port, f'{args.exp_name}_{args.run_id}', args.reload, '.')
    else:
        from common.tensorboard import Visualiser
        args.visualiser = Visualiser(args.log_path, args.exp_name)
    args.device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    args.visualiser.args(args)
    args.func(args)
