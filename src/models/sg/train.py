"""
Code adapted from the StarGAN v2: https://github.com/clovaai/stargan-v2
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from .model import build_model


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        self.optims = Munch()
        for net in self.nets.keys():
            self.optims[net] = torch.optim.Adam(
                params=self.nets[net].parameters(),
                lr=args.f_lr if net == 'mapping_network' else args.lr,
                betas=[args.beta1, args.beta2],
                weight_decay=args.weight_decay)

        self.ckptios = [
            CheckpointIO(ospj(args.model_path, '{:06d}_nets.ckpt'), **self.nets),
            CheckpointIO(ospj(args.model_path, '{:06d}_nets_ema.ckpt'), **self.nets_ema),
            CheckpointIO(ospj(args.model_path, '{:06d}_optims.ckpt'), **self.optims)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the EMA parameters
            if ('ema' not in name):
                print('Initializing %s...' % name)
                network.apply(he_init)

    def _save_checkpoint(self, step, checkpoint):
        for ckptio in self.ckptios:
            ckptio.save(step, checkpoint)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, args.latent_dim, args.device)
        fetcher_val = InputFetcher(loaders.val, args.latent_dim, args.device)
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            lambda_ds = args.lambda_ds * (1 - i / args.total_iters)
            # fetch images and labels
            inputs = next(fetcher)
            x_real, d_org = inputs.x_src, inputs.d_src
            x_trg, x_ds, d_trg = inputs.x_src2, inputs.x_ds, inputs.d_src2
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, d_org, d_trg, z_trg=z_trg)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, d_org, d_trg, x_trg=x_trg)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, d_org, d_trg, lambda_ds, z_trgs=[z_trg, z_trg2])
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, d_org, d_trg, lambda_ds, x_refs=[x_trg, x_ds])
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.save_path, exist_ok=True)
                debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1, checkpoint=args.checkpoint)


def compute_d_loss(nets, args, x_real, d_org, d_trg, z_trg=None, x_trg=None):
    assert (z_trg is None) != (x_trg is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, d_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, d_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_trg, d_trg)

        x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, d_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, d_org, d_trg, lambda_ds, z_trgs=None, x_refs=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ds = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, d_trg)
    else:
        s_trg = nets.style_encoder(x_ref, d_trg)

    x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, d_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, d_org)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, d_trg)
    else:
        s_trg2 = nets.style_encoder(x_ds, d_trg)
    x_fake2 = nets.generator(x_real, s_trg2).detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    s_org = nets.style_encoder(x_real, d_org)
    x_rec = nets.generator(x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
         + args.lambda_cyc * loss_cyc \
         - lambda_ds * loss_ds
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


@torch.no_grad()
def debug_image(nets, args, inputs, step):
    x_src, d_src = inputs.x_src,  inputs.d_src
    x_ref, d_ref = inputs.x_src2, inputs.d_src2

    device = inputs.x_src.device
    N = inputs.x_src.size(0)

    # latent-guided image synthesis
    d_trg_list = [torch.tensor(d).repeat(N).to(device)
                  for d in range(min(args.num_domains, 5))]
    z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
    for psi in [1.0]:
        filename = ospj(args.save_path, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        translate_using_latent(nets, args, x_src, d_trg_list, z_trg_list, psi, filename)


@torch.no_grad()
def translate_using_latent(nets, args, x_src, d_trg_list, z_trg_list, psi, filename):
    N, C, H, W = x_src.size()
    x_concat = [x_src]

    for i, d_trg in enumerate(d_trg_list):
        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, d_trg)
            x_fake = nets.generator(x_src, s_trg)
            x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)


def save_image(x, ncol, filename):
    x = (x+1)/2
    x.clamp_(0, 1)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


class CheckpointIO(object):
    def __init__(self, fname_template, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step, checkpoint):
        fname = self.fname_template.format(step)
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            outdict[name] = module.state_dict()
        torch.save(outdict, fname)
        rmpath = self.fname_template.format(step-checkpoint)
        if os.path.exists(rmpath):
            os.remove(rmpath)

    def load(self, step):
        fname = self.fname_template.format(step)
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu'))
        for name, module in self.module_dict.items():
            module.load_state_dict(module_dict[name])


class InputFetcher:
    def __init__(self, loader, latent_dim, device):
        self.loader = loader
        self.latent_dim = latent_dim
        self.device = device

    def _fetch_inputs(self):
        try:
            x, _, d, x2, x_ds, d2 = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, _, d, x2, x_ds, d2 = next(self.iter)
        return x, d, x2, x_ds, d2

    def __next__(self):
        x, d, x2, x_ds, d2 = self._fetch_inputs()
        z_trg = torch.randn(x.size(0), self.latent_dim)
        z_trg2 = torch.randn(x.size(0), self.latent_dim)
        inputs = Munch(x_src=x, x_src2=x2,
                       x_ds=x_ds, d_src2=d2, d_src=d,
                       z_trg=z_trg, z_trg2=z_trg2)

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})
