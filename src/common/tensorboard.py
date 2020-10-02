from tensorboardX import SummaryWriter
import numpy as np
import os
import json
from common.util import get_args_dict
from torchvision.utils import make_grid
import torch


class Visualiser:
    def __init__(self, log_dir, exp_name):
        path = os.path.join(log_dir, exp_name)
        self.visualiser = SummaryWriter(path)

    def image(self, images, title='', step=0):
        if not self.visualiser:
            return
        if images.min() < 0:
            images = (images + 1) / 2
        nrow = int(len(images)**(1/2))
        img = make_grid(torch.from_numpy(images), nrow)
        self.visualiser.add_image(title, img, step)

    def text(self, text, title='', step=0):
        if not self.visualiser:
            return
        self.visualiser.add_text(title, text, step)

    def args(self, namespace):
        d = get_args_dict(namespace)
        text = json.dumps(d)
        text = text.replace(',', '\n')
        self.text(text, 'Arguments')

    def plot(self, data, title=None, step=0):
        if not self.visualiser:
            return
        self.visualiser.add_scalar(title, data, step)

    def matplotlib(self, fig, title='', step=0):
        fig.canvas.draw()
        s, (width, height) = fig.canvas.print_to_buffer()
        data = np.fromstring(s, dtype=np.uint8, sep='').reshape((height, width, 4))[:, :, :3]
        self.visualiser.add_image(title, data.transpose(2, 0, 1), step)
