import torch
from ..model import Generator, MappingNetwork
import torchvision.utils as vutils
from common.loaders import images


def save_image(x, ncol, filename):
    x = (x + 1) / 2
    x.clamp_(0, 1)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=2, pad_value=1)


def parse_args(parser):
    parser.add_argument('--state-dict-path', type=str, help='Path to the model state dict')
    parser.add_argument('--dataset-src', type=str, help='Dataset in {dataset_mnist, dataset_svhn}')
    parser.add_argument('--data-root-src', type=str, help='Path to the data')
    parser.add_argument('--domain', type=int, help='Domain id {0, 1}')
    parser.add_argument('--img-size', type=int, default=32, help='Size of the image')
    parser.add_argument('--save-name', type=str, help='Name of the sample file')
    parser.add_argument('--bottleneck-size', type=int, default=64)


@torch.no_grad()
def execute(args):
    state_dict_path = args.state_dict_path
    domain = args.domain
    name = args.save_name

    device = 'cuda'
    N = 64
    latent_dim = 16
    domain = int(domain)
    # Load model
    state_dict = torch.load(state_dict_path, map_location='cpu')
    generator = Generator(bottleneck_size=args.bottleneck_size, bottleneck_blocks=4, img_size=args.img_size, max_conv_dim=128).to(device)
    generator.load_state_dict(state_dict['generator'])
    mapping = MappingNetwork()
    mapping.load_state_dict(state_dict['mapping_network'])
    mapping.to(device)

    dataset = getattr(images, args.dataset_src)
    dataset = dataset(args.data_root_src)
    src_dataset = torch.utils.data.DataLoader(dataset, batch_size=N, num_workers=10)

    data = next(iter(src_dataset))
    data = data.to(device)
    print(data.min(), data.max())

    # Infer translated images
    d_trg = torch.tensor(domain).repeat(N).long().to(device)
    z_trg = torch.randn(N, latent_dim).to(device)
    print(z_trg.shape, data.shape)

    x_concat = [data]

    s_trg = mapping(z_trg, d_trg)
    print(data.shape, s_trg.shape)
    print(data.min(), data.max())
    x_fake = generator(data, s_trg)
    x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    results = [None] * len(x_concat)
    results[::2] = x_concat[:len(x_concat)//2]
    results[1::2] = x_concat[len(x_concat)//2:]
    results = torch.stack(results)
    save_image(results, 10, f'{name}.png')
