from .train import train
from common.loaders import images
import torchvision
import torch


def parse_args(parser):
    parser.add_argument('--dataset-loc1', type=str, default='./data/train/real', help='Location dataset 1')
    parser.add_argument('--dataset-loc2', type=str, default='./data/train/sketch', help='location dataset 2')
    parser.add_argument('--dataset1', type=str, default='single_visda', help='Dataset 1')
    parser.add_argument('--dataset2', type=str, default='single_visda', help='Dataset 2')
    parser.add_argument('--h-dim', type=int, default=256, help='Number of hidden Dimensions')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam parameter')

    parser.add_argument('--z-dim', type=int, default=2048, help='Bottleneck size')
    parser.add_argument('--cw', type=float, default=1, help='Lambda source classifier')
    parser.add_argument('--tcw', type=float, default=1, help='Lambda target cluster')
    parser.add_argument('--dw', type=float, default=1, help='Lambda Adversrial')
    parser.add_argument('--smw', type=float, default=1, help='Lambda source manifold')
    parser.add_argument('--tmw', type=float, default=1, help='Lambda target manifold')
    parser.add_argument('--ss-path', type=str, required=True, help='Path of pre-trained self-supervised model (MoCO v2)')


def execute(args):
    print(args)
    dataset1 = getattr(images, args.dataset1)
    dataset2 = getattr(images, args.dataset2)
    train_loader1, test_loader1, shape1, nc = dataset1(
        args.dataset_loc1, args.train_batch_size, args.test_batch_size, normalize=False)
    train_loader2, test_loader2, shape2, _ = dataset2(
        args.dataset_loc2, args.train_batch_size, args.test_batch_size, normalize=False)
    args.loaders1 = (train_loader1, test_loader1)
    args.loaders2 = (train_loader2, test_loader2)
    args.shape1 = shape1
    args.shape2 = shape2
    args.nc = nc

    ssx = torchvision.models.resnet50().to(args.device)
    ssx.fc = torch.nn.Identity()
    state_dict = torch.load(args.ss_path, map_location='cpu')['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    err = ssx.load_state_dict(state_dict, strict=False)
    print(err)
    print(ssx)
    ssx.eval()
    args.ssx = ssx

    train(args)
