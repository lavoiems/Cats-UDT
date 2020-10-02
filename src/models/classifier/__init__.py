# Model inspired from: https://github.com/szagoruyko/wide-residual-networks
      
from .train import train
from common.loaders import images


def parse_args(parser):
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist','svhn','svhn_extra'], help='Dataset to use for training the classifier')
    parser.add_argument('--dataset-loc', type=str, default='.', help='Dataset path')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--depth', type=float, default=16, help='Depth of the wide ResNet')
    parser.add_argument('--widen-factor', type=float, default=8, help='Widen factor of the wide ResNet')
    parser.add_argument('--dropRate', type=float, default=0.4, help='Dropout probability')


def execute(args):
    print(args)
    train_loader, valid_loader, test_loader, shape, nc = \
        getattr(images, args.dataset)(args.dataset_loc, args.train_batch_size, args.test_batch_size, args.valid_split)
    args.nc = nc
    args.loader = (train_loader, valid_loader, test_loader)
    args.shape = shape
    train(args)
