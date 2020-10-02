import numpy as np
import visdom
import os
import json
from common.util import get_args_dict


class Visualiser:
    def __init__(self, server, port, exp_name, reload, visdom_dir):
        if not server or not port:
            self.visualiser = None
            print('No server or no port defined. Visualisation are disabled.')
            return

        log_name = None
        if visdom_dir:
            log_name = os.path.join(visdom_dir, 'visdom.out')

        self.visualiser = visdom.Visdom(
            server, port=port, env=exp_name, use_incoming_socket=False)
        print(exp_name, server, port)
        if not reload:
            self.visualiser.delete_env(exp_name)

    def image(self, images, title=None, step=0):
        if images.min() < 0:
            images = (images + 1) / 2
        nrow = int(len(images)**(1/2))
        if not self.visualiser:
            return
        win = 'images%s%s' % (self.visualiser.env, title)
        self.visualiser.images(images, nrow=nrow, win=win, opts={'title': title}, env=None)

    def text(self, text, identifier=0, env=None):
        if not self.visualiser:
            return
        self.visualiser.text(text, win='text%s' % identifier, env=env)

    def args(self, namespace):
        d = get_args_dict(namespace)
        text = json.dumps(d)
        text = text.replace(',', '<br>')
        self.text(text)

    def plot(self, data, title=None, step=0, update='append'):
        if not self.visualiser:
            return
        win = 'plot%s%s' % (self.visualiser.env, title)
        opts = {'title': title}
        dataY = np.array([data])
        step = np.array([step])
        err = self.visualiser.line(X=step, Y=dataY, win=win, opts=opts, update=update)
        if win != err:
            self.visualiser.line(X=step, Y=data, win=win, opts=opts)

    def video(self, videofile, title=None, step=0):
        if not self.visualiser:
            return
        win = 'video%s%s' % (self.visualiser.env, title)
        opts = {'fps': 1, 'title': title}
        self.visualiser.video(videofile=videofile, win=win, opts=opts)

    def matplotlib(self, fig, title='', step=0):
        fig.canvas.draw()
        s, (width, height) = fig.canvas.print_to_buffer()
        data = np.fromstring(s, dtype=np.uint8, sep='').reshape((height, width, 4))[:, :, :3]
        self.image(data.transpose(2, 0, 1), title, step)
