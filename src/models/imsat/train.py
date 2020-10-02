import time
import torch
import torch.nn.functional as F
from torch import optim
from sklearn.decomposition import PCA
import matplotlib.pylab as plt

from common.util import sample, save_models, one_hot_embedding
from common.initialize import initialize, infer_iteration
from . import model


def gp_loss(x, y, d, device):
    batch_size = x.size()[0]
    gp_alpha = torch.rand(batch_size, 1, device=device)

    interpx = gp_alpha * x.data + (1 - gp_alpha) * y.data
    interpx.requires_grad = True
    d_interp = d(interpx)
    grad_interp = torch.autograd.grad(outputs=d_interp, inputs=(interpx,),
                                      grad_outputs=torch.ones(d_interp.size(), device=device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    diff1 = grad_interp.norm(2, dim=1) - 1
    diff1 = torch.clamp(diff1, 0)
    return torch.mean(diff1 ** 2)


def compute_loss(x, xp, encoder, contrastive, device):
    z = encoder(x)
    zp = encoder(xp)

    p = contrastive(z)
    closs = p.mean()

    dloss = F.mse_loss(zp, z).mean()
    return dloss, closs


def contrastive_loss(x, n_classes, encoder, contrastive, device):
    enc = encoder(x)

    z = torch.randint(n_classes, size=(enc.shape[0],))
    z = one_hot_embedding(z, n_classes).to(device)
    cz = contrastive(z).mean()
    cenc = contrastive(enc).mean()
    gp = gp_loss(enc, z, contrastive, device)
    return cz, cenc, gp


def define_models(shape, **parameters):
    classifier = model.Encoder(shape[0], **parameters)
    contrastive = model.Contrastive(**parameters)
    return {
        'classifier': classifier,
        'contrastive': contrastive,
    }


@torch.no_grad()
def evaluate_clusters(visualiser, encoder, target, label, id):
    enc = encoder(target)
    pca = PCA(2)
    emb = pca.fit_transform(enc.reshape(enc.shape[0], -1).cpu().squeeze().numpy())
    fig = plt.figure()
    colors = [f'C{c}' for c in label.cpu().numpy()]
    plt.scatter(*emb.transpose(), c=colors)
    visualiser.matplotlib(fig, f'Embeddings {id}', None)
    plt.clf()
    plt.close(fig)


@torch.no_grad()
def evaluate_accuracy(visualiser, i, loader, classifier, nlabels, id, device):
    labels = []
    preds = []
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        pred = F.softmax(classifier(data), 1)
        pred = classifier(data)
        labels += [label]
        preds += [pred]
    labels = torch.cat(labels)
    preds = torch.cat(preds).argmax(1)
    correct = 0
    total = 0
    for j in range(nlabels):
        label = labels[preds == j]
        if len(label):
            correct += one_hot_embedding(label, nlabels).sum(0).max()
        total += len(label)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    visualiser.plot(accuracy, title=f'Classifier accuracy {id}', step=i)
    return accuracy


def evaluate(visualiser, data, datap, id):
    visualiser.image(data.cpu().detach().numpy(), f'target{id}', 0)
    visualiser.image(datap.cpu().detach().numpy(), f'target p {id}', 0)


def train(args):
    parameters = vars(args)
    train_loader1, test_loader1 = args.loaders

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    encoder = models['classifier'].to(args.device)
    contrastive = models['contrastive'].to(args.device)
    print(encoder)
    print(contrastive)

    optim_encoder = optim.Adam(encoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_contrastive = optim.Adam(contrastive.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    iter1 = iter(train_loader1)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    mone = torch.FloatTensor([-1]).to(args.device)
    t0 = time.time()
    for i in range(iteration, args.iterations):
        encoder.train()
        contrastive.train()

        for _ in range(args.d_updates):
            batchx, iter1 = sample(iter1, train_loader1)
            datax = batchx[0].float().to(args.device)

            optim_contrastive.zero_grad()
            ploss, nloss, gp = contrastive_loss(datax, args.nc, encoder, contrastive, args.device)
            (ploss - nloss + gp).backward()
            optim_contrastive.step()

        optim_encoder.zero_grad()

        batchx, iter1 = sample(iter1, train_loader1)
        datax = batchx[0].float().to(args.device)
        dataxp = batchx[1].float().to(args.device)

        dloss, closs = compute_loss(datax, dataxp, encoder, contrastive, args.device)
        (args.ld * dloss + closs).backward()
        optim_encoder.step()

        if i % args.evaluate == 0:
            encoder.eval()
            contrastive.eval()
            print('Iter: {}'.format(i), end=': ')
            evaluate(args.visualiser, datax, dataxp, 'x')
            _acc = evaluate_accuracy(args.visualiser, i, test_loader1, encoder, args.nc, 'x', args.device)
            print('disc loss: {}'.format((ploss - nloss).detach().cpu().numpy()), end='\t')
            print('gp: {}'.format(gp.detach().cpu().numpy()), end='\t')
            print('positive dist loss: {}'.format(dloss.detach().cpu().numpy()), end='\t')
            print('contrast. loss: {}'.format(closs.detach().cpu().numpy()), end='\t')
            print('Accuracy: {}'.format(_acc))

            t0 = time.time()
            save_models(models, i, args.model_path, args.checkpoint)
