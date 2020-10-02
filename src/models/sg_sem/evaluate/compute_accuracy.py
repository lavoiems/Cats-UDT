import os
import math
import torch
from ..model import Generator, MappingNetwork, semantics
import torchvision.utils as vutils
from common.loaders import images
from common.initialize import get_last_model
import torch.nn.functional as F
from common.initialize import define_last_model
from common.util import normalize


def evaluate(loader, nz, domain, sem, mapping, generator, classifier, device):
    correct = 0
    total = 0

    for data, label in loader:
        data = data*2 - 1
        N = len(data)
        d_trg = torch.tensor(domain).repeat(N).long().to(device)
        data, label = data.to(device), label.to(device)
        y = sem((data+1)*0.5).argmax(1)

        z = torch.randn(N, nz).to(device)
        s = mapping(z, y, d_trg)
        gen = generator(data, s)

        gen = normalize(gen)
        pred = F.softmax(classifier(gen), 1).argmax(1)
        correct += (pred == label).sum().cpu().float()
        total += len(pred)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    print(accuracy)
    save_image(normalize(data), 'data.png')
    save_image(gen, 'gen.png')
    return accuracy


def save_image(x, filename):
    ncol = int(math.sqrt(len(x)))
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=2, pad_value=1)


def save_result(save_path, identifier, state_dict_path, value):
    filename = os.path.join(save_path, f'accuracy_id:{identifier}.txt')
    with open(filename, 'w') as f:
        f.write(f'{state_dict_path}\n')
        f.write(f'{value}\n')


def parse_args(parser):
    parser.add_argument('--identifier', type=str, required=True, help='Identifier for saving artefact')
    parser.add_argument('--save-path', type=str, help='Path to the trained model')
    parser.add_argument('--classifier-path', type=str, help='Path to the classifier model')
    parser.add_argument('--data-root-src', type=str, help='Path to the data')
    parser.add_argument('--dataset-src', type=str, default='dataset_single', help='name of the dataset in {mnist,svhn}')
    parser.add_argument('--domain', type=int, help='Domain id {0, 1}')
    parser.add_argument('--img-size', type=int, default=32, help='Size of the image')
    parser.add_argument('--max-conv-dim', type=int, default=512)
    parser.add_argument('--bottleneeck-size', type=int, default=64, help='Size of the bottleneck')
    parser.add_argument('--bottleneck_blocks', type=int, default=4, help='Number of layers at the bottleneck')
    parser.add_argument('--da-path', type=str, help='Domain adaptation path')


@torch.no_grad()
def execute(args):
    data_root_src = args.data_root_src
    domain = args.domain
    nz = 16
    save_path = args.save_path
    state_dict_path = get_last_model('nets_ema', save_path)

    device = 'cuda'
    domain = int(domain)
    # Load model
    state_dict = torch.load(state_dict_path, map_location='cpu')
    generator = Generator(bottleneck_size=64, bottleneck_blocks=4, img_size=args.img_size, max_conv_dim=args.max_conv_dim).to(device)
    generator.load_state_dict(state_dict['generator'])
    mapping = MappingNetwork(nc=10)
    mapping.load_state_dict(state_dict['mapping_network'])
    mapping.to(device)

    sem = semantics(None, 'vmt_cluster', args.da_path, shape=[3, 32], nc=10).cuda()
    sem.eval()

    classifier = define_last_model('classifier', args.classifier_path, 'classifier', shape=3, nc=10).to(device)
    classifier.eval()

    dataset = getattr(images, args.dataset_src)
    src_dataset = dataset(data_root_src, 1, 32)[2]

    accuracy = evaluate(src_dataset, nz, domain, sem, mapping, generator, classifier, device)
    print(accuracy)

    save_result(save_path, args.identifier, state_dict_path, accuracy)
