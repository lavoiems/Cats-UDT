from .train import train
from common.loaders import images


def parse_args(parser):
    parser.add_argument('--dataset-loc', type=str, default='./data', help='Location of the dataset')
    parser.add_argument('--dataset', type=str, default='imnist', choices=['imnist', 'isvhn'], help='Dataset to use for training')
    parser.add_argument('--h-dim', type=int, default=512, help='N hidden channels in the network')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0, help='Adam parameter')
    parser.add_argument('--beta2', type=float, default=0.99, help='Adam parameter')
    parser.add_argument('--d-updates', type=int, default=4, help='N critic updates per generator update')
    parser.add_argument('--ld', type=float, default=1, help='Lambda distance loss')


def execute(args):
    print(args)
    dataset = getattr(images, args.dataset)
    train_loader, _, test_loader, shape, n_classes = dataset(
        args.dataset_loc, args.train_batch_size, args.test_batch_size, args.valid_split)
    args.loaders = (train_loader, test_loader)
    args.shape = shape

    args.nc = n_classes

    train(args)
