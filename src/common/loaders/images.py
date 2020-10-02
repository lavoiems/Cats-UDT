import os
import random
import torch.utils.data as data
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler


def svhn(root, train_batch_size, test_batch_size, valid_split=0, **kwargs):
    transform = transforms.Compose((
        transforms.ToTensor(),))
    train = datasets.SVHN(root, split='train', download=True, transform=transform)
    n_classes = len(set(train.labels))
    test = datasets.SVHN(root, split='test', download=True, transform=transform)

    idxes = np.arange(len(train))
    split = int(np.floor(valid_split * len(idxes)))
    train_sampler = SubsetRandomSampler(idxes[split:])
    valid_sampler = SubsetRandomSampler(idxes[:split])
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=train_sampler, num_workers=8, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=valid_sampler, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, num_workers=8, shuffle=False)

    shape = train_loader.dataset[0][0].shape

    return train_loader, valid_loader, test_loader, shape, n_classes


def svhn_extra(root, train_batch_size, test_batch_size, valid_split=0, **kwargs):
    transform = transforms.Compose((
        transforms.ToTensor(),))
    train = datasets.SVHN(root, split='train', download=True, transform=transform)
    n_classes = len(set(train.labels))
    extra = datasets.SVHN(root, split='extra', download=True, transform=transform)
    train = torch.utils.data.ConcatDataset((train, extra))
    test = datasets.SVHN(root, split='test', download=True, transform=transform)

    idxes = np.arange(len(train))
    split = int(np.floor(valid_split * len(idxes)))
    train_sampler = SubsetRandomSampler(idxes[split:])
    valid_sampler = SubsetRandomSampler(idxes[:split])
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=train_sampler, num_workers=8, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=valid_sampler, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, num_workers=8)

    shape = train_loader.dataset[0][0].shape

    return train_loader, valid_loader, test_loader, shape, n_classes


def triple_channel(x):
    if x.shape[0] == 3:
        return x
    return torch.cat((x,x,x), 0)


def mnist(root, train_batch_size, test_batch_size, valid_split=0, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
        triple_channel,
        ])
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)

    idxes = np.arange(len(train))
    split = int(np.floor(valid_split * len(idxes)))
    train_sampler = SubsetRandomSampler(idxes[split:])
    valid_sampler = SubsetRandomSampler(idxes[:split])
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=train_sampler, num_workers=8, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=valid_sampler, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, shuffle=False, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.train_labels.tolist()))

    return train_loader, valid_loader, test_loader, shape, n_classes


def imnist(root, train_batch_size, test_batch_size, valid_split, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
        triple_channel,
    ])
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    n_classes = len(set(train.train_labels.tolist()))
    t = [transforms.RandomAffine(30, (0, 0), (0.5, 1.5), 40), transforms.ToTensor()]
    train = MultiTransformDataset(train, t)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=False, num_workers=8)

    shape = train_loader.dataset[0][0].shape

    return train_loader, test_loader, test_loader, shape, n_classes


@torch.no_grad()
def cond_mnist_svhn(root, train_batch_size, test_batch_size, semantics, nc, device, **kwargs):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
        triple_channel,
        normalize,
    ])

    train1 = datasets.MNIST(root, train=True, download=True, transform=transform)
    train2 = datasets.SVHN(root, split='train', download=True, transform=transform)
    train = CondDataset(train1, train2, semantics, nc, device)
    test1 = datasets.MNIST(root, train=False, download=True, transform=transform)
    test2 = datasets.SVHN(root, split='test', download=True, transform=transform)
    test = CondDataset(test1, test2, semantics, nc, device)

    train_loader = data.DataLoader(train, batch_size=train_batch_size, shuffle=True,
                                   num_workers=10, drop_last=True, pin_memory=False)
    test_loader = data.DataLoader(test, batch_size=test_batch_size, shuffle=True,
                                  num_workers=10, drop_last=False)
    shape = train_loader.dataset[0][0].shape
    return train_loader, test_loader, shape, nc


def mnist_svhn(root, train_batch_size, test_batch_size, **kwargs):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
        triple_channel,
        normalize,
    ])

    train1 = datasets.MNIST(root, train=True, download=True, transform=transform)
    train2 = datasets.SVHN(root, split='train', download=True, transform=transform)
    train = CondDataset(train1, train2)
    test1 = datasets.MNIST(root, train=False, download=True, transform=transform)
    test2 = datasets.SVHN(root, split='test', download=True, transform=transform)
    test = CondDataset(test1, test2)

    train_loader = data.DataLoader(train, batch_size=train_batch_size, shuffle=True,
                                   num_workers=10, drop_last=True, pin_memory=False)
    test_loader = data.DataLoader(test, batch_size=test_batch_size, shuffle=True,
                                  num_workers=10, drop_last=False)
    shape = train_loader.dataset[0][0].shape
    return train_loader, test_loader, shape, None


def single_visda(root, train_batch_size, test_batch_size, shuffle=True, **kwargs):
    crop = transforms.RandomResizedCrop(
        256, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < 0.5 else x)
    train_transform = [
        rand_crop,
        transforms.Resize((256, 256), interpolation=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    test_transform = [
        transforms.Resize((256, 256), interpolation=1),
        transforms.ToTensor(),
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)
    train = datasets.ImageFolder(root, train_transform)
    test = datasets.ImageFolder(root, test_transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, pin_memory=False,
                                               shuffle=shuffle, num_workers=10, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, shuffle=shuffle,
                                              num_workers=10, drop_last=False)

    shape = train_loader.dataset[0][0].shape
    return train_loader, test_loader, shape, 5


def visda(root, train_batch_size, test_batch_size, shuffle=True, **kwargs):
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    crop = transforms.RandomResizedCrop(
        256, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < 0.5 else x)
    train_transform = [
        rand_crop,
        transforms.Resize((256, 256), interpolation=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    test_transform = [
        transforms.Resize((256, 256), interpolation=1),
        transforms.ToTensor(),
        normalize,
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)
    train = SourceDataset(os.path.join(root, 'train'), None, train_transform)
    test = SourceDataset(os.path.join(root, 'test'), None, test_transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, pin_memory=False,
                                               shuffle=shuffle, num_workers=10, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, shuffle=shuffle,
                                              num_workers=10, drop_last=False)

    shape = train_loader.dataset[0][0].shape
    return train_loader, test_loader, shape, 1


@torch.no_grad()
def cond_visda(root, train_batch_size, test_batch_size, semantics, nc, device, **kwargs):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    crop = transforms.RandomResizedCrop(
        256, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < 0.5 else x)
    train_transform = transforms.Compose([
        rand_crop,
        transforms.Resize((256, 256), interpolation=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=1),
        transforms.ToTensor(),
        normalize,
    ])

    train = SourceDataset(os.path.join(root, 'train'), semantics, train_transform)
    test = SourceDataset(os.path.join(root, 'test'), semantics, test_transform)

    train_loader = data.DataLoader(train, batch_size=train_batch_size, shuffle=True,
                                   num_workers=8, drop_last=True, pin_memory=True)
    test_loader = data.DataLoader(test, batch_size=test_batch_size, shuffle=True,
                                  num_workers=8, drop_last=False)
    shape = train_loader.dataset[0][0].shape
    return train_loader, test_loader, shape, nc


class dataset_mnist(data.Dataset):
    def __init__(self, dataroot):
        transform = transforms.Compose([
            transforms.Resize(32, interpolation=0),
            transforms.ToTensor(),
            triple_channel,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = datasets.MNIST(dataroot, train=False, download=True, transform=transform)

    def __getitem__(self, index):
        return self.dataset[index][0]

    def __len__(self):
        return len(self.dataset)


class dataset_svhn(data.Dataset):
    def __init__(self, dataroot):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = datasets.SVHN(dataroot, split='test', download=True, transform=transform)

    def __getitem__(self, index):
        return self.dataset[index][0]

    def __len__(self):
        return len(self.dataset)


class dataset_single(data.Dataset):
    def __init__(self, dataroot):
        self.dataroot = dataroot
        images = os.listdir(self.dataroot)
        self.img = [os.path.join(self.dataroot, x) for x in images]
        self.img = list(sorted(self.img))
        self.size = len(self.img)
        self.input_dim = 3

        # setup image transformation
        transform = [transforms.Resize((256, 256), 1)]
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = transforms.Compose(transform)

    def __getitem__(self, index):
        data = self.load_img(self.img[index])
        return data

    def load_img(self, img_name):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.img)


class SourceDataset(data.Dataset):
    def __init__(self, root, semantic=None, transform=None):
        self.datasets, self.targets, self.domains = self._make_dataset(root, transform, semantic)

        d1 = os.listdir(root)[0]
        n_labels = len(os.listdir(os.path.join(root, d1)))
        self.labels_idxs = [torch.nonzero(self.targets == label)[:, 0] for label in range(n_labels)]

    def _make_dataset(self, root, transform, semantic):
        domain_names = os.listdir(root)
        datas = []
        labels = []
        domains = []
        maps = [1, 0, 4, 2, 3]
        #maps = [2, 3, 0, 1, 4]
        for idx, domain in enumerate(sorted(domain_names)):
            correct = 0
            total = 0
            path = os.path.join(root, domain)
            dataset = datasets.ImageFolder(path, transform)
            if semantic:
                for data, gt in dataset:
                    data = data.cuda()
                    data = (data.unsqueeze(0)+1)*0.5
                    label = semantic(data).argmax(1)
                    labels.append(label)
                    correct += int(maps[gt] == label)
                    total += 1
                print(f'Accuracy for {domain}: {correct / total}')
            else:
                labels += [0] * len(dataset)
            datas.append(dataset)
            domains += [idx] * len(dataset)
        return torch.utils.data.ConcatDataset(datas), torch.LongTensor(labels), domains

    def __getitem__(self, index):
        sample, _ = self.datasets[index]
        target = self.targets[index]
        domain = self.domains[index]
        idxs = self.labels_idxs[target]
        idx2 = idxs[random.randint(0, len(idxs)-1)]

        sample2, target2 = self.datasets[idx2]
        domain2 = self.domains[idx2]

        idx_domain = torch.nonzero(torch.LongTensor(self.domains) == domain2, as_tuple=True)[0]
        idxs_ds = list(set(idxs.tolist()) & set(idx_domain.tolist()))
        idx_ds = idxs_ds[random.randint(0, len(idxs_ds)-1)]
        sample_ds, _ = self.datasets[idx_ds]

        return sample, target, domain, sample2, sample_ds, domain2

    def __len__(self):
        return len(self.datasets)


class CondDataset(data.Dataset):
    def __init__(self, dataset1, dataset2, semantics=None, nc=10, device='cuda'):
        labels = []
        self.domains = [0]*len(dataset1) + [1]*len(dataset2)
        self.dataset = data.ConcatDataset((dataset1, dataset2))
        if semantics:

            print('Infering semantics for dataset1')
            for sample, _ in dataset1:
                sample = sample.to(device)
                sample = (sample.unsqueeze(0)+1)*0.5
                label = semantics(sample).argmax(1)
                labels.append(label)
            print('Infering semantics for dataset2')
            for sample, _ in dataset2:
                sample = sample.to(device)
                sample = (sample.unsqueeze(0)+1)*0.5
                label = semantics(sample).argmax(1)
                labels.append(label)

            self.labels = torch.LongTensor(labels)
            self.labels_idxs = [torch.nonzero(self.labels == label)[:, 0] for label in range(nc)]
        else:
            self.labels = torch.LongTensor([0]*len(self.domains))
            self.labels_idxs = [torch.arange(len(self.labels))]

    def __getitem__(self, idx):
        sample, _ = self.dataset[idx]
        target = self.labels[idx]
        domain = self.domains[idx]
        idxs = self.labels_idxs[target]
        idx2 = idxs[random.randint(0, len(idxs)-1)]

        sample2, _ = self.dataset[idx2]
        domain2 = self.domains[idx2]

        idx_domain = torch.nonzero(torch.LongTensor(self.domains) == domain2, as_tuple=True)[0]
        idxs_ds = list(set(idxs.tolist()) & set(idx_domain.tolist()))
        idx_ds = idxs_ds[random.randint(0, len(idxs_ds)-1)]
        sample_ds, _ = self.dataset[idx_ds]
        return sample, target, domain, sample2, sample_ds, domain2

    def __len__(self):
        return len(self.dataset)


class MultiTransformDataset(data.Dataset):
    def __init__(self, dataset, t):
        self.dataset = dataset
        self.transform = transforms.Compose(
            [transforms.ToPILImage()] +
            t)

    def __getitem__(self, idx):
        input, target = self.dataset[idx]
        return input, self.transform(input), target

    def __len__(self):
        return len(self.dataset)
