# Model inspired form: https://github.com/xudonmao/VMT

from importlib import import_module
from .train import train
from common.loaders import images
from common.util import get_args
from common.initialize import load_last_model


def parse_args(parser):
    parser.add_argument('--dataset-loc1', type=str, default='./data', help='Location dataset 1')
    parser.add_argument('--dataset-loc2', type=str, default='./data', help='location dataset 2')
    parser.add_argument('--dataset1', type=str, default='mnist', choices=['mnist', 'svhn'], help='Dataset 1')
    parser.add_argument('--dataset2', type=str, default='svhn', choices=['mnist', 'svhn'], help='Dataset2')
    parser.add_argument('--h-dim', type=int, default=256, help='Number of hidden Dimensions')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam parameter')

    parser.add_argument('--z_dim', type=int, default=64, help='Bottleneck size')
    parser.add_argument('--radius', type=float, default=3.5, help='VAT radius')
    parser.add_argument('--cw', type=float, default=1, help='Lambda source classifier loss')
    parser.add_argument('--tcw', type=float, default=0.1, help='Lambda target cluster loss')
    parser.add_argument('--dw', type=float, default=0.01, help='Lambda adversarial loss')
    parser.add_argument('--svw', type=float, default=1, help='Lambda source VAT loss')
    parser.add_argument('--tvw', type=float, default=0.1, help='Lambda target VAT loss')
    parser.add_argument('--smw', type=float, default=1, help='Lambda source manifold loss')
    parser.add_argument('--tmw', type=float, default=0.1, help='Lambda target manifold loss')
    parser.add_argument('--cluster-model-path', type=str, required=True, help='Pre-trained cluster model path (imsat)')


def execute(args):
    print(args)
    dataset1 = getattr(images, args.dataset1)
    train_loader1, _, test_loader1, shape, nc = dataset1(
        args.dataset_loc1, args.train_batch_size, args.test_batch_size)
    args.loaders1 = (train_loader1, test_loader1)
    args.shape = shape
    args.nc = nc

    dataset2 = getattr(images, args.dataset2)
    train_loader2, _, test_loader2, shape2, _ = dataset2(
        args.dataset_loc2, args.train_batch_size, args.test_batch_size)
    args.loaders2 = (train_loader2, test_loader2)
    args.shape2 = shape2

    model_definition = import_module('.'.join(('models', 'imsat', 'train')))
    model_parameters = get_args(args.cluster_model_path)
    model_parameters['nc'] = nc
    models = model_definition.define_models(shape, **model_parameters)
    cluster = models['classifier']
    cluster = load_last_model(cluster, 'classifier', args.cluster_model_path)
    args.cluster = cluster


    train(args)
