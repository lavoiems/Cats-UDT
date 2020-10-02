import time
import torch
import torch.nn.functional as F
from torch import optim

from common.util import sample, save_models, one_hot_embedding
from common.initialize import initialize, infer_iteration
from . import model


def soft_cross_entropy(preds, soft_targets):
    return torch.sum(-F.softmax(soft_targets, 1)*F.log_softmax(preds, 1), 1)


def classification_loss(data, label, classifier):
    pred = classifier(data)
    return F.cross_entropy(pred, label)


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


def compute_perturb(x, y, radius, classifier, device):
    eps = 1e-6 * F.normalize(torch.randn_like(x, device=device))
    eps.requires_grad=True
    xe = x + eps
    ye = classifier(xe)
    loss = soft_cross_entropy(ye, y)
    grad = torch.autograd.grad(loss, eps, grad_outputs=torch.ones_like(loss, device=device),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = F.normalize(grad)
    x_prime = radius*grad + x
    return x_prime.detach()


def vat_loss(x, classifier, radius, device):
    y = classifier(x)
    x_prime = compute_perturb(x, y, radius, classifier, device)
    y_prime = classifier(x_prime)
    return soft_cross_entropy(y_prime, y.detach()).mean()


def mixup_loss(x, classifier, device):
    alpha = torch.rand(x.shape[0], device=device)
    alphax = alpha.view(-1, 1, 1, 1)
    alphay = alpha.view(-1, 1)
    idx = torch.randperm(len(x), device=device)
    x2 = x[idx]
    y = classifier(x)
    y2 = y[idx]

    mix_x = alphax*x + (1-alphax)*x2
    mix_y = alphay*y + (1-alphay)*y2

    mix_yp = classifier(mix_x)
    return soft_cross_entropy(mix_yp, mix_y.detach()).mean()


def define_models(shape, **parameters):
    classifier = model.Classifier(shape[0], **parameters)
    discriminator = model.Discriminator(shape[0], shape[1], **parameters)
    return {
        'classifier': classifier,
        'discriminator': discriminator
    }


@torch.no_grad()
def evaluate_cluster_accuracy(visualiser, i, loader, class_map, classifier, id, device):
    correct = 0
    total = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        pred = F.softmax(classifier(data), 1).argmax(1)
        pred = class_map[pred]
        correct += (pred == label).sum().cpu().float()
        total += len(pred)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    visualiser.plot(accuracy, title=f'Clustering accuracy {id}', step=i)
    return accuracy


@torch.no_grad()
def evaluate_gen_class_accuracy(visualiser, i, loader, nz, nc, encoder, classifier, generator, id, device):
    correct = 0
    total = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        z = torch.randn(data.shape[0], nz, device=device)
        l = encoder(data).argmax(1)
        l = one_hot_embedding(l, nc).to(device)
        gen = generator(l, z)
        pred = F.softmax(classifier(gen), 1).argmax(1)
        correct += (pred == label).sum().cpu().float()
        total += len(pred)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    visualiser.plot(accuracy, title=f'Generated accuracy', step=i)
    return accuracy


@torch.no_grad()
def evaluate_class_accuracy(visualiser, i, loader, classifier, device):
    correct = 0
    total = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        pred = F.softmax(classifier(data), 1).argmax(1)
        correct += (pred == label).sum().cpu().float()
        total += len(pred)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    visualiser.plot(accuracy, title=f'Target test accuracy', step=i)
    return accuracy

@torch.no_grad()
def evaluate_cluster(visualiser, i, nc, loader, classifier, id, device):
    labels = []
    preds = []
    n_preds = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        pred = F.softmax(classifier(data), 1)
        labels += [label]
        preds += [pred]
        n_preds += len(pred)
    labels = torch.cat(labels)
    preds = torch.cat(preds).argmax(1)
    correct = 0
    total = 0
    cluster_map = [0 for _ in range(nc)]
    for j in range(nc):
        label = labels[preds == j]
        if len(label):
            l = one_hot_embedding(label, nc).sum(0)
            correct += l.max()
            cluster_map[j] = l.argmax()
        total += len(label)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    visualiser.plot(accuracy, title=f'Transfer clustering accuracy {id}', step=i)
    return torch.LongTensor(cluster_map).to(device)


@torch.no_grad()
def evaluate(visualiser, encoder, nc, data1, target, z_dim, generator, device):
    z = torch.randn(data1.shape[0], z_dim, device=device)
    visualiser.image(data1.cpu().numpy(), 'target1', 0)
    visualiser.image(target.cpu().numpy(), 'target2', 0)
    enc = encoder(data1).argmax(1)
    enc = one_hot_embedding(enc, nc).to(device)
    X = generator(enc, z)
    visualiser.image(X.cpu().numpy(), f'data{id}', 0)

    merged = len(X)*2 * [None]
    merged[:2*len(data1):2] = data1
    merged[1:2*len(X):2] = X
    merged = torch.stack(merged)
    visualiser.image(merged.cpu().numpy(), f'Comparison{id}', 0)

    z = torch.stack(nc*[z[:nc-1]]).transpose(0, 1).reshape(-1, z.shape[1])
    data1 = torch.cat((nc-1)*[data1[:nc]])
    e1 = encoder(data1).argmax(1)
    e1 = one_hot_embedding(e1, nc).to(device)
    X = generator(e1, z)
    X = torch.cat((data1[:nc], X))
    visualiser.image(X.cpu().numpy(), f'Z effect{id}', 0)


def train(args):
    parameters = vars(args)
    train_loader1, test_loader1 = args.loaders1
    train_loader2, test_loader2 = args.loaders2

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    classifier = models['classifier'].to(args.device)
    discriminator = models['discriminator'].to(args.device)
    cluster = args.cluster.eval().to(args.device)
    print(classifier)
    print(discriminator)

    optim_classifier = optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1 = iter(train_loader1)
    iter2 = iter(train_loader2)
    titer1 = iter(test_loader1)
    titer2 = iter(test_loader2)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        classifier.train()
        discriminator.train()
        batchx, iter1 = sample(iter1, train_loader1)
        data1 = batchx[0].to(args.device)
        if data1.shape[0] != args.train_batch_size:
            batchx, iter1 = sample(iter1, train_loader1)
            data1 = batchx[0].to(args.device)
        label = cluster(data1).argmax(1).detach()

        batchy, iter2 = sample(iter2, train_loader2)
        data2 = batchy[0].to(args.device)
        if data2.shape[0] != args.train_batch_size:
            batchy, iter2 = sample(iter2, train_loader2)
            data2 = batchy[0].to(args.device)

        optim_discriminator.zero_grad()
        d_loss = disc_loss(data1, data2, discriminator, classifier.x, classifier.mlp, args.device)
        d_loss.backward()
        optim_discriminator.step()

        optim_classifier.zero_grad()
        c_loss = classification_loss(data1, label, classifier)
        tcw_loss = classification_target_loss(data2, classifier)
        dw_loss = embed_div_loss(data1, data2, discriminator, classifier.x, classifier.mlp, args.device)
        v_loss1 = vat_loss(data1, classifier, args.radius, args.device)
        v_loss2 = vat_loss(data2, classifier, args.radius, args.device)
        m_loss1 = mixup_loss(data1, classifier, args.device)
        m_loss2 = mixup_loss(data2, classifier, args.device)
        (args.cw *c_loss)  .backward()
        (args.tcw*tcw_loss).backward()
        (args.dw *dw_loss) .backward()
        (args.svw*v_loss1) .backward()
        (args.tvw*v_loss2) .backward()
        (args.smw*m_loss1) .backward()
        (args.tmw*m_loss2) .backward()
        optim_classifier.step()

        if i % args.evaluate == 0:
            classifier.eval()
            batchx, titer1 = sample(titer1, test_loader1)
            data1 = batchx[0].to(args.device)
            batchy, titer2 = sample(titer2, test_loader2)
            data2 = batchy[0].to(args.device)
            print('Iter: %s' % i, time.time() - t0)
            class_map = evaluate_cluster(args.visualiser, i, args.nc, test_loader1, cluster, f'x', args.device)
            evaluate_cluster_accuracy(args.visualiser, i, test_loader1, class_map, classifier, f'x', args.device)
            evaluate_cluster_accuracy(args.visualiser, i, test_loader2, class_map, classifier, f'y', args.device)
            args.visualiser.plot(c_loss.cpu().detach().numpy(), title=f'Classifier loss', step=i)
            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
