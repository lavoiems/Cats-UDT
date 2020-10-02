import os
import torch
from ..model import Generator, MappingNetwork, semantics
from torch.utils import data
from common.util import normalize, get_args
from common.initialize import get_last_model
from common.loaders import images
from common.evaluation import fid


def parse_args(parser):
    parser.add_argument('--identifier', type=str, required=True, help='Identifier for saving artefact')
    parser.add_argument('--save-path', type=str, help='Path of the trained model')
    parser.add_argument('--domain', type=int, help='Domain id [0, 1]')
    parser.add_argument('--ss-path', type=str, help='Self-supervised model path')
    parser.add_argument('--da-path', type=str, help='Domain adaptation path')
    parser.add_argument('--model-type', type=str, help='DA model type in {vmt_cluster, vmtc_repr}')
    parser.add_argument('--data-root-src', type=str, help='Path of the data')
    parser.add_argument('--data-root-tgt', type=str, help='Path of the data')
    parser.add_argument('--dataset-src', type=str, help='Dataset in {dataset_single, dataset_mnist, dataset_svhn}')
    parser.add_argument('--dataset-tgt', type=str, help='Dataset in {dataset_single, dataset_mnist, dataset_svhn}')
    parser.add_argument('--img-size', type=int, default=256, help='Size of the image')
    parser.add_argument('--nc', type=int, default=5, help='Number of classes')


def save_result(save_path, identifier, state_dict_path, value):
    filename = os.path.join(save_path, f'fid_id:{identifier}.txt')
    with open(filename, 'w') as f:
        f.write(f'{state_dict_path}\n')
        f.write(f'{value}\n')


@torch.no_grad()
def execute(args):
    device = 'cuda'
    latent_dim = 16
    batch_size = 128
    # Load model
    save_path = args.save_path
    state_dict_path = get_last_model('nets_ema', save_path)
    state_dict = torch.load(state_dict_path, map_location='cpu')

    bottleneck_size = get_args(save_path)['bottleneck_size']
    generator = Generator(bottleneck_size=bottleneck_size, bottleneck_blocks=4, img_size=args.img_size).to(device)
    generator.load_state_dict(state_dict['generator'])
    mapping = MappingNetwork(nc=args.nc)
    mapping.load_state_dict(state_dict['mapping_network'])
    mapping.to(device)

    sem = semantics(args.ss_path, args.model_type, args.da_path, nc=args.nc, shape=[3, args.img_size]).to(device)
    sem.eval()

    dataset = getattr(images, args.dataset_src)(args.data_root_src)
    src = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10)
    dataset = getattr(images, args.dataset_tgt)(args.data_root_tgt)
    trg = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=10)

    print(f'Src size: {len(src)}, Tgt size: {len(trg)}')
    generated = []
    #print('Fetching generated data')
    d = torch.tensor(args.domain).repeat(batch_size).long().to(device)
    for data in src:
        data = data.to(device)
        d_trg = d[:data.shape[0]]
        y_trg = sem((data+1)*0.5).argmax(1)
        for i in range(5):
            z_trg = torch.randn(data.shape[0], latent_dim, device=device)
            s_trg = mapping(z_trg, y_trg, d_trg)
            gen = generator(data, s_trg)
            generated.append(gen)
    generated = torch.cat(generated)
    generated = normalize(generated)
    #save_image(generated[:4], 'Debug.png')

    #print('Fetching target data')
    trg_data = []
    for data in trg:
        data = data.to(device)
        trg_data.append(data)
    trg_data = torch.cat(trg_data)
    #print(trg_data.shape)

    trg_data = normalize(trg_data)
    #print(generated.min(), generated.max(), trg_data.min(), trg_data.max())
    computed_fid = fid.calculate_fid(trg_data, generated, 512, device, 2048)
    print(f'FID: {computed_fid}')
    save_result(save_path, args.identifier, state_dict_path, computed_fid)
