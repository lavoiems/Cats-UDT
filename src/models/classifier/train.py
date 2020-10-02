import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from common.util import save_models, sample, normalize_channels
from common.initialize import initialize, infer_iteration
from . import model


def prior(batch_size, nz, device):
    return torch.randn(batch_size, nz, device=device)


def classification_loss(data, classes, classifier):
    pred = classifier(data)
    return F.cross_entropy(pred, classes)


def define_models(shape, **parameters):
    classifier = model.Classifier(**parameters)
    return {
        'classifier': classifier,
    }


@torch.no_grad()
def evaluate(visualiser, i, loader, classifier, id, device):
    correct = 0
    total = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        pred = F.softmax(classifier(data), 1).argmax(1)
        correct += (pred == label).sum().cpu().float()
        total += len(pred)
    accuracy = correct / total
    accuracy = accuracy.cpu().numpy()
    visualiser.plot(accuracy, title=f'Accuracy {id}', step=i)
    return accuracy


def train(args):
    parameters = vars(args)
    train_loader, valid_loader, test_loader = args.loader

    models = define_models(**parameters)
    initialize(models, args.reload, args.save_path, args.model_path)

    classifier = models['classifier'].to(args.device)
    print(classifier)
    optim_classifier = optim.SGD(classifier.parameters(), lr=args.lr, momentum=args.momentum,
                                 nesterov=args.nesterov, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_classifier, args.iterations)

    it = iter(train_loader)
    iteration = infer_iteration(list(models.keys())[0], args.reload, args.model_path, args.save_path)
    t0 = time.time()
    best_accuracy = 0
    for i in range(iteration, args.iterations):
        classifier.train()
        batch, it = sample(it, train_loader)
        data = batch[0].to(args.device)
        classes = batch[1].to(args.device)

        optim_classifier.zero_grad()
        loss = classification_loss(data, classes, classifier)
        loss.backward()
        optim_classifier.step()
        scheduler.step()

        if i % args.evaluate == 0:
            classifier.eval()
            valid_accuracy = evaluate(args.visualiser, i, valid_loader, classifier, 'valid', args.device)
            test_accuracy = evaluate(args.visualiser, i, test_loader, classifier, 'test', args.device)
            loss = loss.detach().cpu().numpy()
            args.visualiser.plot(loss, title=f'loss', step=i)
            print(f'Iter: {i}\tTime:{time.time()-t0}\tLoss:{loss}\tvalid_accuracy:{valid_accuracy}\ttest_accuracy:{test_accuracy}')
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                save_models(models, i, args.model_path, args.checkpoint)
            t0 = time.time()
