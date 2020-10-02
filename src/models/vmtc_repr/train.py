import os
import time
import torch
import torch.nn.functional as F
from torch import optim
from sklearn.cluster import SpectralClustering

from common.util import one_hot_embedding, save_models
from common.initialize import initialize, infer_iteration
from . import model


def he_init(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)


@torch.no_grad()
def get_initial_zx(loader, ss, device):
    initial_zx = []
    labels = []
    for data, label in loader:
        data = data.to(device)
        zx = ss(data)
        initial_zx += [zx]
        labels += [label.to(device)]
    return torch.cat(initial_zx), torch.cat(labels)


def soft_cross_entropy(preds, soft_targets):
    return torch.sum(-F.softmax(soft_targets, 1)*F.log_softmax(preds, 1), 1)


def classification_loss(data, classes, classifier):
    pred = classifier(data)
    return F.cross_entropy(pred, classes)


def classification_target_loss(data, classifier):
    preds = classifier(data)
    return soft_cross_entropy(preds, preds).mean()


def disc_loss(data1, data2, discriminator, embedding, classifier, device):
    emb1 = embedding(data1).detach()
    emb2 = embedding(data2).detach()
    c1 = classifier(emb1).detach()
    c2 = classifier(emb2).detach()
    pos_dis = discriminator(emb1, c1)
    neg_dis = discriminator(emb2, c2)
    ones = torch.ones_like(pos_dis, device=device)
    zeros = torch.zeros_like(neg_dis, device=device)
    pos_loss = F.binary_cross_entropy_with_logits(pos_dis, ones)
    neg_loss = F.binary_cross_entropy_with_logits(neg_dis, zeros)
    return 0.5*pos_loss + 0.5*neg_loss


def embed_div_loss(data1, data2, discriminator, embedding, classifier, device):
    emb1 = embedding(data1)
    emb2 = embedding(data2)
    c1 = classifier(emb1)
    c2 = classifier(emb2)
    pos_dis = discriminator(emb1, c1)
    neg_dis = discriminator(emb2, c2)
    zeros = torch.zeros_like(pos_dis, device=device)
    ones = torch.ones_like(neg_dis, device=device)
    pos_loss = F.binary_cross_entropy_with_logits(pos_dis, zeros)
    neg_loss = F.binary_cross_entropy_with_logits(neg_dis, ones)
    return 0.5*pos_loss + 0.5*neg_loss


def mixup_loss(x, classifier, device):
    alpha = torch.rand(x.shape[0], device=device)
    alphax = alpha.view(-1, 1)
    alphay = alpha.view(-1, 1)
    idx = torch.randperm(len(x), device=device)
    x2 = x[idx]
    y = classifier(x)
    y2 = y[idx]

    mix_x = alphax*x + (1-alphax)*x2
    mix_y = alphay*y + (1-alphay)*y2

    mix_yp = classifier(mix_x)
    return soft_cross_entropy(mix_yp, mix_y.detach()).mean()


def define_models(**parameters):
    discriminator = model.Discriminator(**parameters)
    classifier = model.Classifier(**parameters)
    return {
        'classifier': classifier,
        'discriminator': discriminator,
    }


@torch.no_grad()
def evaluate_cluster(visualiser, i, nc, zx, labels, classifier, id, device):
    #for data, label in loader:
    data, labels = zx.to(device), labels.to(device)
    preds = F.softmax(classifier(data), 1)
    preds = preds.argmax(1)
    correct = 0
    total = 0
    cluster_map = []
    for j in range(nc):
        label = labels[preds == j]
        if len(label):
            l = one_hot_embedding(label, nc).sum(0)
            correct += l.max()
            cluster_map.append(l.argmax())
        else:
            cluster_map.append(0)
        total += len(label)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    visualiser.plot(accuracy, title=f'Transfer clustering accuracy {id}', step=i)
    return torch.LongTensor(cluster_map).to(device)


@torch.no_grad()
def evaluate_cluster_accuracy(visualiser, i, zs, labels, class_map, classifier, id, device):
    correct = 0
    total = 0

    data, label = zs.to(device), labels.to(device)
    pred = F.softmax(classifier(data), 1).argmax(1)
    pred = class_map[pred]
    correct += (pred == label).sum().cpu().float()
    total += len(pred)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    visualiser.plot(accuracy, title=f'Clustering accuracy {id}', step=i)
    print(f'Accuracy {id}: {accuracy}')
    return accuracy


def train(args):
    parameters = vars(args)
    train_loader1, test_loader1 = args.loaders1
    train_loader2, test_loader2 = args.loaders2

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    ssx = args.ssx.to(args.device)
    ssx.eval()

    zxs, labelsx = get_initial_zx(train_loader1, ssx, args.device)
    zys, labelsy = get_initial_zx(train_loader2, ssx, args.device)

    sc = SpectralClustering(args.nc, affinity='sigmoid', gamma=1.7)
    clusters = sc.fit_predict(zxs.cpu().numpy())
    clusters = torch.from_numpy(clusters).to(args.device)

    classifier = models['classifier'].to(args.device)
    discriminator = models['discriminator'].to(args.device)
    classifier.apply(he_init)
    discriminator.apply(he_init)
    print(classifier)
    print(discriminator)

    optim_discriminator = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_classifier = optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optims = {'optim_discriminator': optim_discriminator, 'optim_classifier': optim_classifier}

    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        classifier.train()
        discriminator.train()

        perm = torch.randperm(len(zxs))
        ix = perm[:args.train_batch_size]
        zx = zxs[ix]
        perm = torch.randperm(len(zys))
        iy = perm[:args.train_batch_size]
        zy = zys[iy]

        optim_discriminator.zero_grad()
        d_loss = disc_loss(zx, zy, discriminator, classifier.x, classifier.mlp, args.device)
        d_loss.backward()
        optim_discriminator.step()

        perm = torch.randperm(len(zxs))
        ix = perm[:args.train_batch_size]
        zx = zxs[ix]
        label = clusters[ix].long()
        perm = torch.randperm(len(zys))
        iy = perm[:args.train_batch_size]
        zy = zys[iy]

        optim_classifier.zero_grad()
        c_loss = classification_loss(zx, label, classifier)
        tcw_loss = classification_target_loss(zy, classifier)
        dw_loss = embed_div_loss(zx, zy, discriminator, classifier.x, classifier.mlp, args.device)
        m_loss1 = mixup_loss(zx, classifier, args.device)
        m_loss2 = mixup_loss(zy, classifier, args.device)
        (args.cw *c_loss)  .backward()
        (args.tcw*tcw_loss).backward()
        (args.dw *dw_loss) .backward()
        (args.smw*m_loss1) .backward()
        (args.tmw*m_loss2) .backward()
        optim_classifier.step()

        if i % args.evaluate == 0:
            print('Iter: %s' % i, time.time() - t0)
            classifier.eval()

            class_map = evaluate_cluster(args.visualiser, i, args.nc, zxs, labelsx, classifier, f'x', args.device)
            evaluate_cluster_accuracy(args.visualiser, i, zxs, labelsx, class_map, classifier, f'x', args.device)
            evaluate_cluster_accuracy(args.visualiser, i, zys, labelsy, class_map, classifier, f'y', args.device)

            save_path = args.save_path
            with open(os.path.join(save_path, 'c_loss'), 'a') as f: f.write(f'{i},{c_loss.cpu().item()}\n')
            with open(os.path.join(save_path, 'tcw_loss'), 'a') as f: f.write(f'{i},{tcw_loss.cpu().item()}\n')
            with open(os.path.join(save_path, 'dw_loss'), 'a') as f: f.write(f'{i},{dw_loss.cpu().item()}\n')
            with open(os.path.join(save_path, 'm_loss1'), 'a') as f: f.write(f'{i},{m_loss1.cpu().item()}\n')
            with open(os.path.join(save_path, 'm_loss2'), 'a') as f: f.write(f'{i},{m_loss2.cpu().item()}\n')
            with open(os.path.join(save_path, 'd_loss2'), 'a') as f: f.write(f'{i},{d_loss.cpu().item()}\n')
            args.visualiser.plot(c_loss.cpu().detach().numpy(), title='Source classifier loss', step=i)
            args.visualiser.plot(tcw_loss.cpu().detach().numpy(), title='Target classifier cross entropy', step=i)
            args.visualiser.plot(dw_loss.cpu().detach().numpy(), title='Classifier marginal divergence', step=i)
            args.visualiser.plot(m_loss1.cpu().detach().numpy(), title='Source mix up loss', step=i)
            args.visualiser.plot(m_loss2.cpu().detach().numpy(), title='Target mix up loss', step=i)
            args.visualiser.plot(d_loss.cpu().detach().numpy(), title='Discriminator loss', step=i)
            t0 = time.time()
            save_models(models, i, args.model_path, args.evaluate)
            save_models(optims, i, args.model_path, args.evaluate)
