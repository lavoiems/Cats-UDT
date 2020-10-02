import torch
from ..model import Generator, MappingNetwork, semantics
import torchvision.utils as vutils
from common.loaders import images


def save_image(x, ncol, filename):
    x = (x + 1) / 2
    x.clamp_(0, 1)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=2, pad_value=1)


def parse_args(parser):
    parser.add_argument('--state-dict-path', type=str, help='Path to the model state dict')
    parser.add_argument('--dataset-src', type=str, help='Name of the dataset')
    parser.add_argument('--data-root-src', type=str, help='Path to the data')
    parser.add_argument('--domain', type=int, help='Domain id {0, 1}')
    parser.add_argument('--img-size', type=int, default=32, help='Size of the image')
    parser.add_argument('--da-path', type=str, help='Domain adaptation path')
    parser.add_argument('--save-name', type=str, help='Name of the sample file')


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
    generator = Generator(bottleneck_size=64, bottleneck_blocks=4, img_size=args.img_size).to(device)
    generator.load_state_dict(state_dict['generator'])
    mapping = MappingNetwork(nc=10)
    mapping.load_state_dict(state_dict['mapping_network'])
    mapping.to(device)

    sem = semantics(None, 'vmt_cluster', args.da_path, shape=[3, 32], nc=10).cuda()
    sem.eval()

    dataset = getattr(images, args.dataset_src)
    src_dataset = dataset(args.data_root_src, 1, N)[2]

    data, labels = next(iter(src_dataset))
    data = data.to(device)
    print(data.min(), data.max())

    data = data * 2 - 1
    y_src = sem((data+1)*0.5).argmax(1)

    # Infer translated images
    d_trg = torch.tensor(domain).repeat(N).long().to(device)
    z_trg = torch.randn(N, latent_dim).to(device)
    print(z_trg.shape, data.shape, y_src.shape)

    x_concat = [data]

    print(z_trg.shape, y_src.shape, d_trg.shape)
    s_trg = mapping(z_trg, y_src, d_trg)
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


    N = 10
    R = 5
    data = torch.cat([data[:N]]*R, 0)
    y_trg = torch.cat([y_src[:N]]*R, 0)
    d_trg = torch.tensor(domain).repeat(N*R).long().to(device)
    z_trg = torch.randn(R, latent_dim).to(device)
    z_trg = [torch.stack([z]*N) for z in z_trg]
    z_trg = torch.cat(z_trg, 0)
    print(z_trg.shape, y_trg.shape, d_trg.shape)
    s_trg = mapping(z_trg, y_trg, d_trg)
    x_fake = generator(data, s_trg)
    x_concat = torch.cat((data[:N], x_fake), 0)
    save_image(x_concat, 10, f'{name}_z.png')

