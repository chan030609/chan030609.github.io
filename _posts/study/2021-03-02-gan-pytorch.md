---
layout: post
title: Generative Adversarial Netowrk by Ian Goodfellow - PyTorch Implementation
description: >
    A generative adversarial network (GAN) is a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014. It was initially proposed in a paper uploaded at arXiv. In this study, the GAN algorithm was implemented from scratch by using the auto-grad features in PyTorch.
hide_last_modified: true
image: /assets/img/study/2021-03-02-gan-pytorch/1.jpg
category: [study]
tag: [paper-implementation]
---

Ian Goodfellow, my favorite Deep Learning researcher, became the first to uncover the beauty of Generative Adversarial Netowkrs (GANs), an innovative framework using two networks with opposite goals simultaneously to reach the same goal.

Qutoing from the abstract,
> "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake..."

Ian Goodfellow explicitly calls this a "minimax two-player game." The nature of this network, in which one attemtps to maximize the loss while the other to minimize the loss, makes me imagine a Cop and Robber game, where a cop (C) attempts to catch a robber (R). In this network, the cop would be a discriminator (D) whereas the robber would be a generator (G). Both trying their best, the network would reach the midpoint, where G and D achieving 50:50. 

As Goodfellow puts it,
> "... a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation."

The last statement reminds me that this innovative approach is eventually a deep learning architecture, where forward propagation and back propagation occur as normal. We would need two different networks whose losses are defined by using BCELoss, or Binary Cross Entropy.

View the pdf version of the research paper [here](https://arxiv.org/pdf/1406.2661.pdf).

&nbsp;

## Technologies Used

Instead of Tensorflow, PyTorch was selected as an autograd Python framework due to its simplicity and flexibility in algorithm implementations. For the GPU calculation, Google Colab was used instead of AWS or Azure because the MNIST dataset used in the research is small and easily accessible. Other libraries include `argparse`, `os`, `NumPy`, and `math`.

&nbsp;

## Code

The implementation begins here:

### Importing Libraries
First, libraries were imported.
```py
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
```
```py
os.makedirs("images", exist_ok=True)
```
### Designing a Parser

Parsing is processing a Python program and converting the codes into machine language. `argparse` is a Python module that makes it easy to write command-line interfaces.

```py
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
```
```
Namespace(b1=0.5, b2=0.999, batch_size=64, channels=1, fff='/root/.local/share/jupyter/runtime/kernel-049e03f6-b997-4bc3-8163-e1564744c2c1.json', img_size=28, latent_dim=100, lr=0.0002, n_cpu=8, n_epochs=200, sample_interval=400)
```

The last argument added to the parser, as explained above, is a dummy argument. Python needs this argument to execute the code without an error.

### GPU Setup

The following is a list comprehension statement for setting up GPU. The study used one provided by Google Colab Notebook.
```py
cuda = True if torch.cuda.is_available() else False
```

### Generator

```py
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
```

### Discriminator

```py
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
```

### Loss

```py
adversarial_loss = torch.nn.BCELoss()
```

### Model Initialization
```py
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
```

### Data Loader
```py
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
```

### Optimizers
```py
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
```

### Training
The last step of this implementation is designing a training function and checking the results.

Just as it was stated in the paper, hyperparameters were set 62 epochs and 938 inputs per batch. The training process took about an hour and a half. 

```py
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        optimizer_G.zero_grad()

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        gen_imgs = generator(z)

        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
```
```
Streaming output truncated to the last epoch.
[Epoch 199/200] [Batch 0/938] [D loss: 0.176175] [G loss: 2.866865]
[Epoch 199/200] [Batch 1/938] [D loss: 0.247907] [G loss: 2.098035]
[Epoch 199/200] [Batch 2/938] [D loss: 0.259020] [G loss: 1.716152]
[Epoch 199/200] [Batch 3/938] [D loss: 0.192824] [G loss: 2.768083]
[Epoch 199/200] [Batch 4/938] [D loss: 0.171858] [G loss: 2.798156]
[Epoch 199/200] [Batch 5/938] [D loss: 0.251129] [G loss: 2.101049]
[Epoch 199/200] [Batch 6/938] [D loss: 0.204405] [G loss: 2.364502]
[Epoch 199/200] [Batch 7/938] [D loss: 0.173937] [G loss: 2.996769]
[Epoch 199/200] [Batch 8/938] [D loss: 0.200920] [G loss: 2.153268]
[Epoch 199/200] [Batch 9/938] [D loss: 0.195667] [G loss: 2.674259]
[Epoch 199/200] [Batch 10/938] [D loss: 0.224364] [G loss: 2.946456]
[Epoch 199/200] [Batch 11/938] [D loss: 0.261539] [G loss: 2.770864]
[Epoch 199/200] [Batch 12/938] [D loss: 0.287802] [G loss: 2.123888]
[Epoch 199/200] [Batch 13/938] [D loss: 0.210274] [G loss: 2.185224]
[Epoch 199/200] [Batch 14/938] [D loss: 0.228424] [G loss: 2.607241]
[Epoch 199/200] [Batch 15/938] [D loss: 0.235137] [G loss: 2.663824]
[Epoch 199/200] [Batch 16/938] [D loss: 0.241081] [G loss: 2.248631]
[Epoch 199/200] [Batch 17/938] [D loss: 0.211838] [G loss: 2.155747]
[Epoch 199/200] [Batch 18/938] [D loss: 0.238392] [G loss: 2.538156]
[Epoch 199/200] [Batch 19/938] [D loss: 0.228572] [G loss: 2.044291]
[Epoch 199/200] [Batch 20/938] [D loss: 0.288933] [G loss: 2.455524]
[Epoch 199/200] [Batch 21/938] [D loss: 0.201834] [G loss: 2.481862]
[Epoch 199/200] [Batch 22/938] [D loss: 0.226440] [G loss: 2.822574]
[Epoch 199/200] [Batch 23/938] [D loss: 0.281796] [G loss: 1.947769]
[Epoch 199/200] [Batch 24/938] [D loss: 0.302905] [G loss: 3.486054]
[Epoch 199/200] [Batch 25/938] [D loss: 0.294154] [G loss: 1.526298]
[Epoch 199/200] [Batch 26/938] [D loss: 0.406009] [G loss: 3.541315]
[Epoch 199/200] [Batch 27/938] [D loss: 0.235322] [G loss: 2.165184]
[Epoch 199/200] [Batch 28/938] [D loss: 0.166604] [G loss: 3.381012]
[Epoch 199/200] [Batch 29/938] [D loss: 0.202432] [G loss: 3.123329]
[Epoch 199/200] [Batch 30/938] [D loss: 0.197651] [G loss: 2.401347]
[Epoch 199/200] [Batch 31/938] [D loss: 0.149131] [G loss: 2.699328]
[Epoch 199/200] [Batch 32/938] [D loss: 0.312051] [G loss: 3.147662]
[Epoch 199/200] [Batch 33/938] [D loss: 0.319374] [G loss: 2.084843]
[Epoch 199/200] [Batch 34/938] [D loss: 0.182770] [G loss: 2.353233]
[Epoch 199/200] [Batch 35/938] [D loss: 0.340370] [G loss: 3.394494]
[Epoch 199/200] [Batch 36/938] [D loss: 0.266980] [G loss: 1.946675]
[Epoch 199/200] [Batch 37/938] [D loss: 0.216924] [G loss: 2.053276]
[Epoch 199/200] [Batch 38/938] [D loss: 0.144576] [G loss: 2.725845]
[Epoch 199/200] [Batch 39/938] [D loss: 0.192096] [G loss: 2.520464]
[Epoch 199/200] [Batch 40/938] [D loss: 0.190875] [G loss: 2.185779]
[Epoch 199/200] [Batch 41/938] [D loss: 0.198998] [G loss: 2.815124]
[Epoch 199/200] [Batch 42/938] [D loss: 0.271284] [G loss: 2.742115]
[Epoch 199/200] [Batch 43/938] [D loss: 0.196655] [G loss: 1.850389]
[Epoch 199/200] [Batch 44/938] [D loss: 0.203455] [G loss: 2.733541]
[Epoch 199/200] [Batch 45/938] [D loss: 0.212427] [G loss: 2.076927]
[Epoch 199/200] [Batch 46/938] [D loss: 0.265060] [G loss: 2.339022]
[Epoch 199/200] [Batch 47/938] [D loss: 0.141261] [G loss: 1.888652]
[Epoch 199/200] [Batch 48/938] [D loss: 0.180550] [G loss: 3.207644]
[Epoch 199/200] [Batch 49/938] [D loss: 0.193945] [G loss: 2.811635]
[Epoch 199/200] [Batch 50/938] [D loss: 0.239986] [G loss: 1.947300]
[Epoch 199/200] [Batch 51/938] [D loss: 0.176689] [G loss: 2.678433]
[Epoch 199/200] [Batch 52/938] [D loss: 0.209429] [G loss: 2.783323]
[Epoch 199/200] [Batch 53/938] [D loss: 0.154960] [G loss: 2.074146]
[Epoch 199/200] [Batch 54/938] [D loss: 0.178993] [G loss: 2.679138]
[Epoch 199/200] [Batch 55/938] [D loss: 0.156986] [G loss: 2.805498]
[Epoch 199/200] [Batch 56/938] [D loss: 0.277102] [G loss: 2.323480]
[Epoch 199/200] [Batch 57/938] [D loss: 0.212541] [G loss: 2.699544]
[Epoch 199/200] [Batch 58/938] [D loss: 0.142258] [G loss: 2.750588]
[Epoch 199/200] [Batch 59/938] [D loss: 0.223319] [G loss: 2.364267]
[Epoch 199/200] [Batch 60/938] [D loss: 0.207949] [G loss: 2.262400]
[Epoch 199/200] [Batch 61/938] [D loss: 0.170591] [G loss: 2.500728]
[Epoch 199/200] [Batch 62/938] [D loss: 0.200448] [G loss: 2.584990]
[Epoch 199/200] [Batch 63/938] [D loss: 0.190456] [G loss: 2.383467]
[Epoch 199/200] [Batch 64/938] [D loss: 0.225167] [G loss: 2.705842]
[Epoch 199/200] [Batch 65/938] [D loss: 0.240392] [G loss: 2.007471]
[Epoch 199/200] [Batch 66/938] [D loss: 0.137882] [G loss: 2.905286]
[Epoch 199/200] [Batch 67/938] [D loss: 0.140914] [G loss: 3.414638]
[Epoch 199/200] [Batch 68/938] [D loss: 0.161027] [G loss: 2.924945]
[Epoch 199/200] [Batch 69/938] [D loss: 0.269787] [G loss: 1.705006]
[Epoch 199/200] [Batch 70/938] [D loss: 0.435687] [G loss: 3.779987]
[Epoch 199/200] [Batch 71/938] [D loss: 0.251429] [G loss: 2.105175]
[Epoch 199/200] [Batch 72/938] [D loss: 0.234122] [G loss: 1.905600]
[Epoch 199/200] [Batch 73/938] [D loss: 0.203756] [G loss: 3.434660]
[Epoch 199/200] [Batch 74/938] [D loss: 0.332092] [G loss: 2.782169]
[Epoch 199/200] [Batch 75/938] [D loss: 0.290855] [G loss: 1.309071]
[Epoch 199/200] [Batch 76/938] [D loss: 0.172939] [G loss: 3.807981]
[Epoch 199/200] [Batch 77/938] [D loss: 0.124072] [G loss: 3.325928]
[Epoch 199/200] [Batch 78/938] [D loss: 0.187276] [G loss: 2.720728]
[Epoch 199/200] [Batch 79/938] [D loss: 0.240551] [G loss: 2.025061]
[Epoch 199/200] [Batch 80/938] [D loss: 0.117268] [G loss: 2.697719]
[Epoch 199/200] [Batch 81/938] [D loss: 0.161010] [G loss: 3.138935]
[Epoch 199/200] [Batch 82/938] [D loss: 0.193427] [G loss: 1.978319]
[Epoch 199/200] [Batch 83/938] [D loss: 0.213102] [G loss: 3.079724]
[Epoch 199/200] [Batch 84/938] [D loss: 0.183075] [G loss: 2.259280]
[Epoch 199/200] [Batch 85/938] [D loss: 0.126048] [G loss: 2.416496]
[Epoch 199/200] [Batch 86/938] [D loss: 0.203212] [G loss: 2.640033]
[Epoch 199/200] [Batch 87/938] [D loss: 0.160135] [G loss: 2.065248]
[Epoch 199/200] [Batch 88/938] [D loss: 0.193999] [G loss: 3.242598]
[Epoch 199/200] [Batch 89/938] [D loss: 0.270772] [G loss: 2.289585]
[Epoch 199/200] [Batch 90/938] [D loss: 0.227518] [G loss: 2.030389]
[Epoch 199/200] [Batch 91/938] [D loss: 0.335348] [G loss: 2.927154]
[Epoch 199/200] [Batch 92/938] [D loss: 0.220643] [G loss: 2.512608]
[Epoch 199/200] [Batch 93/938] [D loss: 0.150401] [G loss: 2.469673]
[Epoch 199/200] [Batch 94/938] [D loss: 0.240336] [G loss: 2.961619]
[Epoch 199/200] [Batch 95/938] [D loss: 0.222742] [G loss: 2.215324]
[Epoch 199/200] [Batch 96/938] [D loss: 0.241180] [G loss: 3.417722]
[Epoch 199/200] [Batch 97/938] [D loss: 0.199925] [G loss: 2.168583]
[Epoch 199/200] [Batch 98/938] [D loss: 0.244436] [G loss: 3.398481]
[Epoch 199/200] [Batch 99/938] [D loss: 0.240573] [G loss: 2.064991]
[Epoch 199/200] [Batch 100/938] [D loss: 0.216448] [G loss: 3.405599]
[Epoch 199/200] [Batch 101/938] [D loss: 0.204322] [G loss: 2.300185]
[Epoch 199/200] [Batch 102/938] [D loss: 0.331165] [G loss: 2.095814]
[Epoch 199/200] [Batch 103/938] [D loss: 0.147626] [G loss: 2.718173]
[Epoch 199/200] [Batch 104/938] [D loss: 0.239951] [G loss: 2.908235]
[Epoch 199/200] [Batch 105/938] [D loss: 0.240233] [G loss: 2.169338]
[Epoch 199/200] [Batch 106/938] [D loss: 0.157518] [G loss: 3.652269]
[Epoch 199/200] [Batch 107/938] [D loss: 0.338727] [G loss: 3.615009]
[Epoch 199/200] [Batch 108/938] [D loss: 0.249356] [G loss: 2.786917]
[Epoch 199/200] [Batch 109/938] [D loss: 0.250694] [G loss: 2.171416]
[Epoch 199/200] [Batch 110/938] [D loss: 0.184093] [G loss: 3.381763]
[Epoch 199/200] [Batch 111/938] [D loss: 0.240656] [G loss: 2.872394]
[Epoch 199/200] [Batch 112/938] [D loss: 0.277814] [G loss: 2.369463]
[Epoch 199/200] [Batch 113/938] [D loss: 0.187937] [G loss: 3.049331]
[Epoch 199/200] [Batch 114/938] [D loss: 0.219290] [G loss: 2.613178]
[Epoch 199/200] [Batch 115/938] [D loss: 0.221238] [G loss: 1.746080]
[Epoch 199/200] [Batch 116/938] [D loss: 0.208773] [G loss: 2.943754]
[Epoch 199/200] [Batch 117/938] [D loss: 0.237489] [G loss: 2.235196]
[Epoch 199/200] [Batch 118/938] [D loss: 0.182727] [G loss: 3.004667]
[Epoch 199/200] [Batch 119/938] [D loss: 0.187548] [G loss: 2.445139]
[Epoch 199/200] [Batch 120/938] [D loss: 0.214456] [G loss: 2.251993]
[Epoch 199/200] [Batch 121/938] [D loss: 0.174702] [G loss: 2.499389]
[Epoch 199/200] [Batch 122/938] [D loss: 0.233468] [G loss: 3.357507]
[Epoch 199/200] [Batch 123/938] [D loss: 0.192287] [G loss: 1.933854]
[Epoch 199/200] [Batch 124/938] [D loss: 0.160408] [G loss: 2.838576]
[Epoch 199/200] [Batch 125/938] [D loss: 0.233780] [G loss: 2.568066]
[Epoch 199/200] [Batch 126/938] [D loss: 0.225324] [G loss: 2.109516]
[Epoch 199/200] [Batch 127/938] [D loss: 0.158963] [G loss: 2.886057]
[Epoch 199/200] [Batch 128/938] [D loss: 0.117246] [G loss: 2.308040]
[Epoch 199/200] [Batch 129/938] [D loss: 0.283705] [G loss: 2.826630]
[Epoch 199/200] [Batch 130/938] [D loss: 0.207170] [G loss: 1.782422]
[Epoch 199/200] [Batch 131/938] [D loss: 0.232829] [G loss: 3.077774]
[Epoch 199/200] [Batch 132/938] [D loss: 0.218062] [G loss: 3.231408]
[Epoch 199/200] [Batch 133/938] [D loss: 0.322838] [G loss: 1.746397]
[Epoch 199/200] [Batch 134/938] [D loss: 0.270832] [G loss: 3.100635]
[Epoch 199/200] [Batch 135/938] [D loss: 0.237048] [G loss: 2.865094]
[Epoch 199/200] [Batch 136/938] [D loss: 0.256786] [G loss: 2.066683]
[Epoch 199/200] [Batch 137/938] [D loss: 0.174498] [G loss: 2.243693]
[Epoch 199/200] [Batch 138/938] [D loss: 0.173456] [G loss: 2.570050]
[Epoch 199/200] [Batch 139/938] [D loss: 0.307140] [G loss: 2.696197]
[Epoch 199/200] [Batch 140/938] [D loss: 0.247449] [G loss: 1.563818]
[Epoch 199/200] [Batch 141/938] [D loss: 0.317460] [G loss: 3.443931]
[Epoch 199/200] [Batch 142/938] [D loss: 0.271416] [G loss: 1.853646]
[Epoch 199/200] [Batch 143/938] [D loss: 0.253560] [G loss: 2.197430]
[Epoch 199/200] [Batch 144/938] [D loss: 0.138963] [G loss: 2.919649]
[Epoch 199/200] [Batch 145/938] [D loss: 0.383950] [G loss: 2.598378]
[Epoch 199/200] [Batch 146/938] [D loss: 0.375326] [G loss: 1.597615]
[Epoch 199/200] [Batch 147/938] [D loss: 0.212220] [G loss: 2.069392]
[Epoch 199/200] [Batch 148/938] [D loss: 0.363517] [G loss: 3.007132]
[Epoch 199/200] [Batch 149/938] [D loss: 0.232816] [G loss: 2.271666]
[Epoch 199/200] [Batch 150/938] [D loss: 0.268207] [G loss: 2.354819]
[Epoch 199/200] [Batch 151/938] [D loss: 0.275464] [G loss: 2.810034]
[Epoch 199/200] [Batch 152/938] [D loss: 0.300581] [G loss: 2.456253]
[Epoch 199/200] [Batch 153/938] [D loss: 0.269149] [G loss: 1.672925]
[Epoch 199/200] [Batch 154/938] [D loss: 0.340559] [G loss: 3.739594]
[Epoch 199/200] [Batch 155/938] [D loss: 0.246139] [G loss: 2.423197]
[Epoch 199/200] [Batch 156/938] [D loss: 0.246726] [G loss: 1.861832]
[Epoch 199/200] [Batch 157/938] [D loss: 0.219163] [G loss: 2.953902]
[Epoch 199/200] [Batch 158/938] [D loss: 0.519311] [G loss: 3.208576]
[Epoch 199/200] [Batch 159/938] [D loss: 0.557671] [G loss: 0.835201]
[Epoch 199/200] [Batch 160/938] [D loss: 0.367628] [G loss: 5.684066]
[Epoch 199/200] [Batch 161/938] [D loss: 0.230873] [G loss: 5.713280]
[Epoch 199/200] [Batch 162/938] [D loss: 0.236006] [G loss: 3.106361]
[Epoch 199/200] [Batch 163/938] [D loss: 0.456944] [G loss: 0.918808]
[Epoch 199/200] [Batch 164/938] [D loss: 0.332274] [G loss: 5.206392]
[Epoch 199/200] [Batch 165/938] [D loss: 0.171178] [G loss: 4.847824]
[Epoch 199/200] [Batch 166/938] [D loss: 0.149886] [G loss: 2.889167]
[Epoch 199/200] [Batch 167/938] [D loss: 0.213831] [G loss: 1.798667]
[Epoch 199/200] [Batch 168/938] [D loss: 0.097944] [G loss: 4.446108]
[Epoch 199/200] [Batch 169/938] [D loss: 0.228358] [G loss: 4.648345]
[Epoch 199/200] [Batch 170/938] [D loss: 0.221317] [G loss: 2.160997]
[Epoch 199/200] [Batch 171/938] [D loss: 0.148572] [G loss: 2.960673]
[Epoch 199/200] [Batch 172/938] [D loss: 0.145271] [G loss: 3.001099]
[Epoch 199/200] [Batch 173/938] [D loss: 0.230118] [G loss: 2.343754]
[Epoch 199/200] [Batch 174/938] [D loss: 0.156127] [G loss: 2.139822]
[Epoch 199/200] [Batch 175/938] [D loss: 0.117191] [G loss: 2.317348]
[Epoch 199/200] [Batch 176/938] [D loss: 0.180117] [G loss: 3.268355]
[Epoch 199/200] [Batch 177/938] [D loss: 0.152195] [G loss: 2.914780]
[Epoch 199/200] [Batch 178/938] [D loss: 0.258001] [G loss: 2.659580]
[Epoch 199/200] [Batch 179/938] [D loss: 0.234966] [G loss: 1.702999]
[Epoch 199/200] [Batch 180/938] [D loss: 0.150825] [G loss: 3.194473]
[Epoch 199/200] [Batch 181/938] [D loss: 0.199217] [G loss: 3.079361]
[Epoch 199/200] [Batch 182/938] [D loss: 0.239466] [G loss: 2.125591]
[Epoch 199/200] [Batch 183/938] [D loss: 0.271915] [G loss: 1.939311]
[Epoch 199/200] [Batch 184/938] [D loss: 0.221306] [G loss: 2.945332]
[Epoch 199/200] [Batch 185/938] [D loss: 0.214217] [G loss: 2.076486]
[Epoch 199/200] [Batch 186/938] [D loss: 0.249345] [G loss: 2.690402]
[Epoch 199/200] [Batch 187/938] [D loss: 0.219488] [G loss: 1.941745]
[Epoch 199/200] [Batch 188/938] [D loss: 0.233196] [G loss: 2.719275]
[Epoch 199/200] [Batch 189/938] [D loss: 0.136115] [G loss: 2.173642]
[Epoch 199/200] [Batch 190/938] [D loss: 0.126644] [G loss: 3.067119]
[Epoch 199/200] [Batch 191/938] [D loss: 0.200018] [G loss: 2.745654]
[Epoch 199/200] [Batch 192/938] [D loss: 0.212052] [G loss: 1.617399]
[Epoch 199/200] [Batch 193/938] [D loss: 0.153680] [G loss: 3.183761]
[Epoch 199/200] [Batch 194/938] [D loss: 0.145738] [G loss: 2.369671]
[Epoch 199/200] [Batch 195/938] [D loss: 0.116691] [G loss: 2.839876]
[Epoch 199/200] [Batch 196/938] [D loss: 0.217033] [G loss: 2.861600]
[Epoch 199/200] [Batch 197/938] [D loss: 0.280020] [G loss: 1.734772]
[Epoch 199/200] [Batch 198/938] [D loss: 0.219322] [G loss: 3.349952]
[Epoch 199/200] [Batch 199/938] [D loss: 0.255422] [G loss: 3.353108]
[Epoch 199/200] [Batch 200/938] [D loss: 0.351328] [G loss: 2.078959]
[Epoch 199/200] [Batch 201/938] [D loss: 0.296410] [G loss: 1.962892]
[Epoch 199/200] [Batch 202/938] [D loss: 0.299731] [G loss: 3.450152]
[Epoch 199/200] [Batch 203/938] [D loss: 0.299340] [G loss: 2.202378]
[Epoch 199/200] [Batch 204/938] [D loss: 0.205170] [G loss: 2.002729]
[Epoch 199/200] [Batch 205/938] [D loss: 0.102345] [G loss: 3.375744]
[Epoch 199/200] [Batch 206/938] [D loss: 0.264748] [G loss: 3.013491]
[Epoch 199/200] [Batch 207/938] [D loss: 0.242842] [G loss: 1.823845]
[Epoch 199/200] [Batch 208/938] [D loss: 0.238435] [G loss: 2.966769]
[Epoch 199/200] [Batch 209/938] [D loss: 0.264214] [G loss: 2.594947]
[Epoch 199/200] [Batch 210/938] [D loss: 0.212747] [G loss: 2.256713]
[Epoch 199/200] [Batch 211/938] [D loss: 0.195752] [G loss: 2.018966]
[Epoch 199/200] [Batch 212/938] [D loss: 0.153919] [G loss: 3.378415]
[Epoch 199/200] [Batch 213/938] [D loss: 0.197428] [G loss: 2.943531]
[Epoch 199/200] [Batch 214/938] [D loss: 0.248362] [G loss: 2.371788]
[Epoch 199/200] [Batch 215/938] [D loss: 0.158515] [G loss: 2.839327]
[Epoch 199/200] [Batch 216/938] [D loss: 0.326753] [G loss: 3.516816]
[Epoch 199/200] [Batch 217/938] [D loss: 0.248734] [G loss: 1.874995]
[Epoch 199/200] [Batch 218/938] [D loss: 0.326136] [G loss: 2.870335]
[Epoch 199/200] [Batch 219/938] [D loss: 0.164958] [G loss: 2.484247]
[Epoch 199/200] [Batch 220/938] [D loss: 0.155785] [G loss: 3.188093]
[Epoch 199/200] [Batch 221/938] [D loss: 0.283479] [G loss: 3.329817]
[Epoch 199/200] [Batch 222/938] [D loss: 0.364380] [G loss: 1.475576]
[Epoch 199/200] [Batch 223/938] [D loss: 0.187580] [G loss: 3.184144]
[Epoch 199/200] [Batch 224/938] [D loss: 0.276370] [G loss: 3.485568]
[Epoch 199/200] [Batch 225/938] [D loss: 0.300458] [G loss: 2.620525]
[Epoch 199/200] [Batch 226/938] [D loss: 0.243189] [G loss: 1.517256]
[Epoch 199/200] [Batch 227/938] [D loss: 0.177980] [G loss: 3.266880]
[Epoch 199/200] [Batch 228/938] [D loss: 0.209479] [G loss: 3.116762]
[Epoch 199/200] [Batch 229/938] [D loss: 0.285092] [G loss: 1.858942]
[Epoch 199/200] [Batch 230/938] [D loss: 0.287977] [G loss: 3.050403]
[Epoch 199/200] [Batch 231/938] [D loss: 0.252462] [G loss: 2.075593]
[Epoch 199/200] [Batch 232/938] [D loss: 0.178366] [G loss: 2.006252]
[Epoch 199/200] [Batch 233/938] [D loss: 0.205732] [G loss: 3.515969]
[Epoch 199/200] [Batch 234/938] [D loss: 0.170753] [G loss: 2.900948]
[Epoch 199/200] [Batch 235/938] [D loss: 0.183896] [G loss: 2.380811]
[Epoch 199/200] [Batch 236/938] [D loss: 0.296149] [G loss: 2.369880]
[Epoch 199/200] [Batch 237/938] [D loss: 0.268228] [G loss: 2.097771]
[Epoch 199/200] [Batch 238/938] [D loss: 0.251951] [G loss: 2.209613]
[Epoch 199/200] [Batch 239/938] [D loss: 0.181100] [G loss: 2.206546]
[Epoch 199/200] [Batch 240/938] [D loss: 0.207395] [G loss: 3.157980]
[Epoch 199/200] [Batch 241/938] [D loss: 0.181780] [G loss: 2.015794]
[Epoch 199/200] [Batch 242/938] [D loss: 0.112618] [G loss: 2.968054]
[Epoch 199/200] [Batch 243/938] [D loss: 0.191019] [G loss: 2.403472]
[Epoch 199/200] [Batch 244/938] [D loss: 0.278627] [G loss: 2.103539]
[Epoch 199/200] [Batch 245/938] [D loss: 0.225700] [G loss: 2.044623]
[Epoch 199/200] [Batch 246/938] [D loss: 0.157096] [G loss: 2.424322]
[Epoch 199/200] [Batch 247/938] [D loss: 0.241520] [G loss: 3.086684]
[Epoch 199/200] [Batch 248/938] [D loss: 0.156671] [G loss: 1.992711]
[Epoch 199/200] [Batch 249/938] [D loss: 0.159008] [G loss: 2.925520]
[Epoch 199/200] [Batch 250/938] [D loss: 0.224920] [G loss: 2.759626]
[Epoch 199/200] [Batch 251/938] [D loss: 0.231499] [G loss: 1.449306]
[Epoch 199/200] [Batch 252/938] [D loss: 0.224217] [G loss: 3.609385]
[Epoch 199/200] [Batch 253/938] [D loss: 0.203658] [G loss: 2.549701]
[Epoch 199/200] [Batch 254/938] [D loss: 0.194876] [G loss: 1.676840]
[Epoch 199/200] [Batch 255/938] [D loss: 0.135244] [G loss: 2.773341]
[Epoch 199/200] [Batch 256/938] [D loss: 0.211392] [G loss: 3.598412]
[Epoch 199/200] [Batch 257/938] [D loss: 0.224708] [G loss: 2.488272]
[Epoch 199/200] [Batch 258/938] [D loss: 0.269359] [G loss: 1.527549]
[Epoch 199/200] [Batch 259/938] [D loss: 0.200305] [G loss: 2.988211]
[Epoch 199/200] [Batch 260/938] [D loss: 0.250329] [G loss: 2.862903]
[Epoch 199/200] [Batch 261/938] [D loss: 0.269140] [G loss: 1.893545]
[Epoch 199/200] [Batch 262/938] [D loss: 0.163886] [G loss: 2.216187]
[Epoch 199/200] [Batch 263/938] [D loss: 0.259003] [G loss: 3.176598]
[Epoch 199/200] [Batch 264/938] [D loss: 0.247816] [G loss: 2.068399]
[Epoch 199/200] [Batch 265/938] [D loss: 0.201962] [G loss: 2.298328]
[Epoch 199/200] [Batch 266/938] [D loss: 0.173755] [G loss: 2.559587]
[Epoch 199/200] [Batch 267/938] [D loss: 0.290013] [G loss: 3.266735]
[Epoch 199/200] [Batch 268/938] [D loss: 0.240412] [G loss: 1.986515]
[Epoch 199/200] [Batch 269/938] [D loss: 0.109149] [G loss: 2.738789]
[Epoch 199/200] [Batch 270/938] [D loss: 0.255844] [G loss: 3.579661]
[Epoch 199/200] [Batch 271/938] [D loss: 0.155274] [G loss: 2.278192]
[Epoch 199/200] [Batch 272/938] [D loss: 0.236170] [G loss: 3.086045]
[Epoch 199/200] [Batch 273/938] [D loss: 0.352186] [G loss: 2.370769]
[Epoch 199/200] [Batch 274/938] [D loss: 0.228008] [G loss: 1.838458]
[Epoch 199/200] [Batch 275/938] [D loss: 0.255264] [G loss: 3.867733]
[Epoch 199/200] [Batch 276/938] [D loss: 0.327582] [G loss: 2.569546]
[Epoch 199/200] [Batch 277/938] [D loss: 0.136153] [G loss: 2.373286]
[Epoch 199/200] [Batch 278/938] [D loss: 0.258456] [G loss: 4.033010]
[Epoch 199/200] [Batch 279/938] [D loss: 0.205228] [G loss: 2.317050]
[Epoch 199/200] [Batch 280/938] [D loss: 0.213996] [G loss: 2.487602]
[Epoch 199/200] [Batch 281/938] [D loss: 0.135304] [G loss: 3.049213]
[Epoch 199/200] [Batch 282/938] [D loss: 0.148976] [G loss: 2.082105]
[Epoch 199/200] [Batch 283/938] [D loss: 0.155315] [G loss: 3.220047]
[Epoch 199/200] [Batch 284/938] [D loss: 0.201225] [G loss: 3.025921]
[Epoch 199/200] [Batch 285/938] [D loss: 0.202983] [G loss: 2.065297]
[Epoch 199/200] [Batch 286/938] [D loss: 0.098216] [G loss: 2.725986]
[Epoch 199/200] [Batch 287/938] [D loss: 0.176831] [G loss: 3.859935]
[Epoch 199/200] [Batch 288/938] [D loss: 0.171672] [G loss: 2.830805]
[Epoch 199/200] [Batch 289/938] [D loss: 0.184434] [G loss: 2.805463]
[Epoch 199/200] [Batch 290/938] [D loss: 0.254344] [G loss: 2.975509]
[Epoch 199/200] [Batch 291/938] [D loss: 0.291210] [G loss: 2.584691]
[Epoch 199/200] [Batch 292/938] [D loss: 0.351356] [G loss: 1.881077]
[Epoch 199/200] [Batch 293/938] [D loss: 0.229376] [G loss: 3.473210]
[Epoch 199/200] [Batch 294/938] [D loss: 0.242234] [G loss: 2.362588]
[Epoch 199/200] [Batch 295/938] [D loss: 0.180427] [G loss: 2.340036]
[Epoch 199/200] [Batch 296/938] [D loss: 0.244608] [G loss: 3.323972]
[Epoch 199/200] [Batch 297/938] [D loss: 0.197567] [G loss: 2.013613]
[Epoch 199/200] [Batch 298/938] [D loss: 0.213182] [G loss: 2.904261]
[Epoch 199/200] [Batch 299/938] [D loss: 0.225277] [G loss: 2.643142]
[Epoch 199/200] [Batch 300/938] [D loss: 0.208721] [G loss: 1.870620]
[Epoch 199/200] [Batch 301/938] [D loss: 0.261529] [G loss: 3.529472]
[Epoch 199/200] [Batch 302/938] [D loss: 0.224374] [G loss: 2.577248]
[Epoch 199/200] [Batch 303/938] [D loss: 0.303146] [G loss: 1.590743]
[Epoch 199/200] [Batch 304/938] [D loss: 0.266378] [G loss: 5.308264]
[Epoch 199/200] [Batch 305/938] [D loss: 0.174748] [G loss: 4.234241]
[Epoch 199/200] [Batch 306/938] [D loss: 0.237348] [G loss: 2.252128]
[Epoch 199/200] [Batch 307/938] [D loss: 0.223224] [G loss: 2.797141]
[Epoch 199/200] [Batch 308/938] [D loss: 0.303564] [G loss: 2.617024]
[Epoch 199/200] [Batch 309/938] [D loss: 0.236638] [G loss: 3.166992]
[Epoch 199/200] [Batch 310/938] [D loss: 0.165589] [G loss: 2.550902]
[Epoch 199/200] [Batch 311/938] [D loss: 0.133821] [G loss: 2.282986]
[Epoch 199/200] [Batch 312/938] [D loss: 0.258197] [G loss: 3.808869]
[Epoch 199/200] [Batch 313/938] [D loss: 0.291760] [G loss: 1.588293]
[Epoch 199/200] [Batch 314/938] [D loss: 0.408132] [G loss: 3.474592]
[Epoch 199/200] [Batch 315/938] [D loss: 0.299519] [G loss: 1.466373]
[Epoch 199/200] [Batch 316/938] [D loss: 0.200556] [G loss: 2.601952]
[Epoch 199/200] [Batch 317/938] [D loss: 0.171046] [G loss: 2.687467]
[Epoch 199/200] [Batch 318/938] [D loss: 0.243585] [G loss: 2.459690]
[Epoch 199/200] [Batch 319/938] [D loss: 0.301971] [G loss: 1.473862]
[Epoch 199/200] [Batch 320/938] [D loss: 0.350662] [G loss: 3.982964]
[Epoch 199/200] [Batch 321/938] [D loss: 0.332438] [G loss: 2.393125]
[Epoch 199/200] [Batch 322/938] [D loss: 0.386411] [G loss: 1.585981]
[Epoch 199/200] [Batch 323/938] [D loss: 0.518817] [G loss: 4.037899]
[Epoch 199/200] [Batch 324/938] [D loss: 0.371658] [G loss: 1.116412]
[Epoch 199/200] [Batch 325/938] [D loss: 0.245988] [G loss: 3.252386]
[Epoch 199/200] [Batch 326/938] [D loss: 0.323216] [G loss: 2.665028]
[Epoch 199/200] [Batch 327/938] [D loss: 0.238581] [G loss: 2.912842]
[Epoch 199/200] [Batch 328/938] [D loss: 0.302390] [G loss: 1.474202]
[Epoch 199/200] [Batch 329/938] [D loss: 0.248074] [G loss: 3.511686]
[Epoch 199/200] [Batch 330/938] [D loss: 0.203089] [G loss: 2.666372]
[Epoch 199/200] [Batch 331/938] [D loss: 0.199740] [G loss: 1.939941]
[Epoch 199/200] [Batch 332/938] [D loss: 0.246498] [G loss: 2.405722]
[Epoch 199/200] [Batch 333/938] [D loss: 0.184473] [G loss: 2.582722]
[Epoch 199/200] [Batch 334/938] [D loss: 0.311763] [G loss: 2.069279]
[Epoch 199/200] [Batch 335/938] [D loss: 0.216584] [G loss: 1.725886]
[Epoch 199/200] [Batch 336/938] [D loss: 0.159415] [G loss: 2.935909]
[Epoch 199/200] [Batch 337/938] [D loss: 0.184457] [G loss: 2.678483]
[Epoch 199/200] [Batch 338/938] [D loss: 0.175692] [G loss: 2.359285]
[Epoch 199/200] [Batch 339/938] [D loss: 0.129458] [G loss: 2.391854]
[Epoch 199/200] [Batch 340/938] [D loss: 0.177482] [G loss: 2.200067]
[Epoch 199/200] [Batch 341/938] [D loss: 0.242835] [G loss: 2.577452]
[Epoch 199/200] [Batch 342/938] [D loss: 0.199162] [G loss: 2.003941]
[Epoch 199/200] [Batch 343/938] [D loss: 0.153344] [G loss: 2.524630]
[Epoch 199/200] [Batch 344/938] [D loss: 0.121477] [G loss: 3.148416]
[Epoch 199/200] [Batch 345/938] [D loss: 0.162254] [G loss: 2.223209]
[Epoch 199/200] [Batch 346/938] [D loss: 0.198330] [G loss: 2.872133]
[Epoch 199/200] [Batch 347/938] [D loss: 0.260797] [G loss: 2.263832]
[Epoch 199/200] [Batch 348/938] [D loss: 0.169911] [G loss: 2.062124]
[Epoch 199/200] [Batch 349/938] [D loss: 0.174950] [G loss: 2.446357]
[Epoch 199/200] [Batch 350/938] [D loss: 0.238970] [G loss: 3.146230]
[Epoch 199/200] [Batch 351/938] [D loss: 0.252210] [G loss: 1.882972]
[Epoch 199/200] [Batch 352/938] [D loss: 0.165039] [G loss: 2.199133]
[Epoch 199/200] [Batch 353/938] [D loss: 0.184804] [G loss: 3.282881]
[Epoch 199/200] [Batch 354/938] [D loss: 0.152160] [G loss: 2.357124]
[Epoch 199/200] [Batch 355/938] [D loss: 0.209948] [G loss: 2.122189]
[Epoch 199/200] [Batch 356/938] [D loss: 0.242265] [G loss: 2.360388]
[Epoch 199/200] [Batch 357/938] [D loss: 0.210174] [G loss: 2.709091]
[Epoch 199/200] [Batch 358/938] [D loss: 0.163455] [G loss: 2.188987]
[Epoch 199/200] [Batch 359/938] [D loss: 0.169496] [G loss: 2.522282]
[Epoch 199/200] [Batch 360/938] [D loss: 0.146255] [G loss: 2.443640]
[Epoch 199/200] [Batch 361/938] [D loss: 0.158732] [G loss: 2.886055]
[Epoch 199/200] [Batch 362/938] [D loss: 0.154152] [G loss: 2.511513]
[Epoch 199/200] [Batch 363/938] [D loss: 0.178091] [G loss: 2.716209]
[Epoch 199/200] [Batch 364/938] [D loss: 0.134054] [G loss: 2.470751]
[Epoch 199/200] [Batch 365/938] [D loss: 0.240790] [G loss: 2.505597]
[Epoch 199/200] [Batch 366/938] [D loss: 0.228456] [G loss: 2.131683]
[Epoch 199/200] [Batch 367/938] [D loss: 0.205889] [G loss: 2.159759]
[Epoch 199/200] [Batch 368/938] [D loss: 0.126466] [G loss: 3.090704]
[Epoch 199/200] [Batch 369/938] [D loss: 0.206596] [G loss: 3.050979]
[Epoch 199/200] [Batch 370/938] [D loss: 0.315421] [G loss: 1.785874]
[Epoch 199/200] [Batch 371/938] [D loss: 0.205003] [G loss: 2.089457]
[Epoch 199/200] [Batch 372/938] [D loss: 0.304602] [G loss: 4.140142]
[Epoch 199/200] [Batch 373/938] [D loss: 0.210082] [G loss: 2.545969]
[Epoch 199/200] [Batch 374/938] [D loss: 0.242288] [G loss: 1.517764]
[Epoch 199/200] [Batch 375/938] [D loss: 0.332051] [G loss: 3.267748]
[Epoch 199/200] [Batch 376/938] [D loss: 0.167965] [G loss: 2.263455]
[Epoch 199/200] [Batch 377/938] [D loss: 0.138848] [G loss: 2.446100]
[Epoch 199/200] [Batch 378/938] [D loss: 0.212023] [G loss: 2.779009]
[Epoch 199/200] [Batch 379/938] [D loss: 0.201181] [G loss: 1.983401]
[Epoch 199/200] [Batch 380/938] [D loss: 0.210439] [G loss: 2.385793]
[Epoch 199/200] [Batch 381/938] [D loss: 0.197349] [G loss: 2.314613]
[Epoch 199/200] [Batch 382/938] [D loss: 0.133679] [G loss: 2.420263]
[Epoch 199/200] [Batch 383/938] [D loss: 0.212295] [G loss: 2.799887]
[Epoch 199/200] [Batch 384/938] [D loss: 0.145982] [G loss: 2.733064]
[Epoch 199/200] [Batch 385/938] [D loss: 0.234447] [G loss: 2.665583]
[Epoch 199/200] [Batch 386/938] [D loss: 0.197636] [G loss: 2.148088]
[Epoch 199/200] [Batch 387/938] [D loss: 0.264492] [G loss: 2.761209]
[Epoch 199/200] [Batch 388/938] [D loss: 0.154716] [G loss: 2.287759]
[Epoch 199/200] [Batch 389/938] [D loss: 0.218611] [G loss: 4.166057]
[Epoch 199/200] [Batch 390/938] [D loss: 0.173427] [G loss: 2.634358]
[Epoch 199/200] [Batch 391/938] [D loss: 0.246204] [G loss: 1.935047]
[Epoch 199/200] [Batch 392/938] [D loss: 0.116775] [G loss: 2.992719]
[Epoch 199/200] [Batch 393/938] [D loss: 0.295306] [G loss: 3.493567]
[Epoch 199/200] [Batch 394/938] [D loss: 0.295120] [G loss: 1.677125]
[Epoch 199/200] [Batch 395/938] [D loss: 0.139685] [G loss: 2.522015]
[Epoch 199/200] [Batch 396/938] [D loss: 0.162882] [G loss: 2.933967]
[Epoch 199/200] [Batch 397/938] [D loss: 0.142589] [G loss: 2.533268]
[Epoch 199/200] [Batch 398/938] [D loss: 0.143259] [G loss: 2.731997]
[Epoch 199/200] [Batch 399/938] [D loss: 0.217292] [G loss: 2.136044]
[Epoch 199/200] [Batch 400/938] [D loss: 0.159131] [G loss: 2.491596]
[Epoch 199/200] [Batch 401/938] [D loss: 0.227853] [G loss: 2.733583]
[Epoch 199/200] [Batch 402/938] [D loss: 0.226800] [G loss: 1.736660]
[Epoch 199/200] [Batch 403/938] [D loss: 0.261958] [G loss: 3.597746]
[Epoch 199/200] [Batch 404/938] [D loss: 0.191723] [G loss: 2.023865]
[Epoch 199/200] [Batch 405/938] [D loss: 0.204057] [G loss: 2.936778]
[Epoch 199/200] [Batch 406/938] [D loss: 0.233341] [G loss: 2.547791]
[Epoch 199/200] [Batch 407/938] [D loss: 0.144822] [G loss: 2.922557]
[Epoch 199/200] [Batch 408/938] [D loss: 0.202072] [G loss: 2.866778]
[Epoch 199/200] [Batch 409/938] [D loss: 0.257182] [G loss: 2.823847]
[Epoch 199/200] [Batch 410/938] [D loss: 0.149708] [G loss: 2.049132]
[Epoch 199/200] [Batch 411/938] [D loss: 0.211676] [G loss: 3.729592]
[Epoch 199/200] [Batch 412/938] [D loss: 0.141932] [G loss: 3.016027]
[Epoch 199/200] [Batch 413/938] [D loss: 0.297054] [G loss: 2.872088]
[Epoch 199/200] [Batch 414/938] [D loss: 0.245419] [G loss: 1.941454]
[Epoch 199/200] [Batch 415/938] [D loss: 0.387231] [G loss: 3.781829]
[Epoch 199/200] [Batch 416/938] [D loss: 0.212774] [G loss: 1.908115]
[Epoch 199/200] [Batch 417/938] [D loss: 0.250502] [G loss: 3.069856]
[Epoch 199/200] [Batch 418/938] [D loss: 0.209168] [G loss: 1.790280]
[Epoch 199/200] [Batch 419/938] [D loss: 0.164701] [G loss: 3.443392]
[Epoch 199/200] [Batch 420/938] [D loss: 0.266300] [G loss: 4.043994]
[Epoch 199/200] [Batch 421/938] [D loss: 0.312919] [G loss: 2.612050]
[Epoch 199/200] [Batch 422/938] [D loss: 0.249251] [G loss: 1.319867]
[Epoch 199/200] [Batch 423/938] [D loss: 0.465307] [G loss: 5.756533]
[Epoch 199/200] [Batch 424/938] [D loss: 0.213012] [G loss: 3.914158]
[Epoch 199/200] [Batch 425/938] [D loss: 0.301910] [G loss: 1.589379]
[Epoch 199/200] [Batch 426/938] [D loss: 0.173090] [G loss: 2.824761]
[Epoch 199/200] [Batch 427/938] [D loss: 0.188067] [G loss: 3.695261]
[Epoch 199/200] [Batch 428/938] [D loss: 0.117241] [G loss: 2.516989]
[Epoch 199/200] [Batch 429/938] [D loss: 0.216527] [G loss: 2.990263]
[Epoch 199/200] [Batch 430/938] [D loss: 0.141076] [G loss: 2.181615]
[Epoch 199/200] [Batch 431/938] [D loss: 0.079271] [G loss: 2.771192]
[Epoch 199/200] [Batch 432/938] [D loss: 0.163576] [G loss: 3.821588]
[Epoch 199/200] [Batch 433/938] [D loss: 0.207102] [G loss: 3.087691]
[Epoch 199/200] [Batch 434/938] [D loss: 0.167585] [G loss: 2.082575]
[Epoch 199/200] [Batch 435/938] [D loss: 0.177249] [G loss: 2.487347]
[Epoch 199/200] [Batch 436/938] [D loss: 0.165294] [G loss: 3.097478]
[Epoch 199/200] [Batch 437/938] [D loss: 0.181609] [G loss: 2.862927]
[Epoch 199/200] [Batch 438/938] [D loss: 0.192922] [G loss: 2.429098]
[Epoch 199/200] [Batch 439/938] [D loss: 0.171153] [G loss: 2.079747]
[Epoch 199/200] [Batch 440/938] [D loss: 0.154005] [G loss: 3.203406]
[Epoch 199/200] [Batch 441/938] [D loss: 0.152138] [G loss: 2.986746]
[Epoch 199/200] [Batch 442/938] [D loss: 0.157036] [G loss: 2.475485]
[Epoch 199/200] [Batch 443/938] [D loss: 0.127224] [G loss: 2.736212]
[Epoch 199/200] [Batch 444/938] [D loss: 0.213415] [G loss: 2.871835]
[Epoch 199/200] [Batch 445/938] [D loss: 0.241517] [G loss: 2.659885]
[Epoch 199/200] [Batch 446/938] [D loss: 0.199841] [G loss: 1.686439]
[Epoch 199/200] [Batch 447/938] [D loss: 0.202613] [G loss: 3.846711]
[Epoch 199/200] [Batch 448/938] [D loss: 0.163802] [G loss: 3.160515]
[Epoch 199/200] [Batch 449/938] [D loss: 0.166813] [G loss: 1.929572]
[Epoch 199/200] [Batch 450/938] [D loss: 0.125947] [G loss: 3.291141]
[Epoch 199/200] [Batch 451/938] [D loss: 0.194483] [G loss: 3.136602]
[Epoch 199/200] [Batch 452/938] [D loss: 0.131646] [G loss: 2.021608]
[Epoch 199/200] [Batch 453/938] [D loss: 0.079194] [G loss: 3.264104]
[Epoch 199/200] [Batch 454/938] [D loss: 0.176758] [G loss: 3.167364]
[Epoch 199/200] [Batch 455/938] [D loss: 0.195511] [G loss: 2.367797]
[Epoch 199/200] [Batch 456/938] [D loss: 0.397595] [G loss: 2.764567]
[Epoch 199/200] [Batch 457/938] [D loss: 0.368808] [G loss: 1.093210]
[Epoch 199/200] [Batch 458/938] [D loss: 0.172814] [G loss: 4.299794]
[Epoch 199/200] [Batch 459/938] [D loss: 0.145072] [G loss: 4.392745]
[Epoch 199/200] [Batch 460/938] [D loss: 0.199428] [G loss: 2.565401]
[Epoch 199/200] [Batch 461/938] [D loss: 0.198332] [G loss: 2.037874]
[Epoch 199/200] [Batch 462/938] [D loss: 0.187989] [G loss: 2.927374]
[Epoch 199/200] [Batch 463/938] [D loss: 0.151766] [G loss: 2.653996]
[Epoch 199/200] [Batch 464/938] [D loss: 0.208205] [G loss: 2.964540]
[Epoch 199/200] [Batch 465/938] [D loss: 0.179434] [G loss: 2.157233]
[Epoch 199/200] [Batch 466/938] [D loss: 0.264165] [G loss: 2.731572]
[Epoch 199/200] [Batch 467/938] [D loss: 0.183958] [G loss: 2.038239]
[Epoch 199/200] [Batch 468/938] [D loss: 0.144163] [G loss: 2.712668]
[Epoch 199/200] [Batch 469/938] [D loss: 0.211556] [G loss: 3.226909]
[Epoch 199/200] [Batch 470/938] [D loss: 0.223584] [G loss: 1.931439]
[Epoch 199/200] [Batch 471/938] [D loss: 0.223790] [G loss: 2.820353]
[Epoch 199/200] [Batch 472/938] [D loss: 0.181542] [G loss: 2.493112]
[Epoch 199/200] [Batch 473/938] [D loss: 0.114742] [G loss: 2.554905]
[Epoch 199/200] [Batch 474/938] [D loss: 0.107901] [G loss: 3.390590]
[Epoch 199/200] [Batch 475/938] [D loss: 0.143555] [G loss: 3.039629]
[Epoch 199/200] [Batch 476/938] [D loss: 0.161102] [G loss: 2.231187]
[Epoch 199/200] [Batch 477/938] [D loss: 0.216064] [G loss: 3.010992]
[Epoch 199/200] [Batch 478/938] [D loss: 0.260971] [G loss: 1.406570]
[Epoch 199/200] [Batch 479/938] [D loss: 0.154681] [G loss: 3.335321]
[Epoch 199/200] [Batch 480/938] [D loss: 0.109214] [G loss: 3.268330]
[Epoch 199/200] [Batch 481/938] [D loss: 0.151232] [G loss: 2.174694]
[Epoch 199/200] [Batch 482/938] [D loss: 0.149287] [G loss: 2.392927]
[Epoch 199/200] [Batch 483/938] [D loss: 0.352485] [G loss: 3.503790]
[Epoch 199/200] [Batch 484/938] [D loss: 0.243805] [G loss: 1.438169]
[Epoch 199/200] [Batch 485/938] [D loss: 0.126539] [G loss: 4.105576]
[Epoch 199/200] [Batch 486/938] [D loss: 0.307576] [G loss: 4.555952]
[Epoch 199/200] [Batch 487/938] [D loss: 0.173676] [G loss: 2.420696]
[Epoch 199/200] [Batch 488/938] [D loss: 0.136089] [G loss: 2.157163]
[Epoch 199/200] [Batch 489/938] [D loss: 0.197976] [G loss: 3.445804]
[Epoch 199/200] [Batch 490/938] [D loss: 0.176674] [G loss: 3.481160]
[Epoch 199/200] [Batch 491/938] [D loss: 0.265417] [G loss: 2.454640]
[Epoch 199/200] [Batch 492/938] [D loss: 0.279640] [G loss: 1.436370]
[Epoch 199/200] [Batch 493/938] [D loss: 0.212081] [G loss: 4.570658]
[Epoch 199/200] [Batch 494/938] [D loss: 0.202018] [G loss: 4.253188]
[Epoch 199/200] [Batch 495/938] [D loss: 0.240850] [G loss: 2.695001]
[Epoch 199/200] [Batch 496/938] [D loss: 0.220374] [G loss: 2.205655]
[Epoch 199/200] [Batch 497/938] [D loss: 0.245859] [G loss: 3.117461]
[Epoch 199/200] [Batch 498/938] [D loss: 0.278917] [G loss: 2.068410]
[Epoch 199/200] [Batch 499/938] [D loss: 0.274101] [G loss: 2.075534]
[Epoch 199/200] [Batch 500/938] [D loss: 0.212346] [G loss: 2.526062]
[Epoch 199/200] [Batch 501/938] [D loss: 0.155112] [G loss: 2.926640]
[Epoch 199/200] [Batch 502/938] [D loss: 0.182273] [G loss: 2.391161]
[Epoch 199/200] [Batch 503/938] [D loss: 0.198234] [G loss: 3.038626]
[Epoch 199/200] [Batch 504/938] [D loss: 0.164866] [G loss: 2.875840]
[Epoch 199/200] [Batch 505/938] [D loss: 0.165694] [G loss: 2.906575]
[Epoch 199/200] [Batch 506/938] [D loss: 0.255150] [G loss: 3.148186]
[Epoch 199/200] [Batch 507/938] [D loss: 0.179065] [G loss: 2.347207]
[Epoch 199/200] [Batch 508/938] [D loss: 0.141748] [G loss: 2.459480]
[Epoch 199/200] [Batch 509/938] [D loss: 0.159015] [G loss: 3.027733]
[Epoch 199/200] [Batch 510/938] [D loss: 0.164527] [G loss: 2.411961]
[Epoch 199/200] [Batch 511/938] [D loss: 0.102912] [G loss: 3.601050]
[Epoch 199/200] [Batch 512/938] [D loss: 0.196432] [G loss: 3.202413]
[Epoch 199/200] [Batch 513/938] [D loss: 0.195932] [G loss: 2.900516]
[Epoch 199/200] [Batch 514/938] [D loss: 0.150358] [G loss: 3.067508]
[Epoch 199/200] [Batch 515/938] [D loss: 0.174759] [G loss: 3.298049]
[Epoch 199/200] [Batch 516/938] [D loss: 0.188719] [G loss: 2.352930]
[Epoch 199/200] [Batch 517/938] [D loss: 0.170931] [G loss: 3.408147]
[Epoch 199/200] [Batch 518/938] [D loss: 0.252167] [G loss: 4.407849]
[Epoch 199/200] [Batch 519/938] [D loss: 0.244339] [G loss: 1.775046]
[Epoch 199/200] [Batch 520/938] [D loss: 0.217820] [G loss: 4.618347]
[Epoch 199/200] [Batch 521/938] [D loss: 0.260512] [G loss: 3.279960]
[Epoch 199/200] [Batch 522/938] [D loss: 0.417691] [G loss: 1.388159]
[Epoch 199/200] [Batch 523/938] [D loss: 0.262616] [G loss: 5.024324]
[Epoch 199/200] [Batch 524/938] [D loss: 0.134651] [G loss: 3.713538]
[Epoch 199/200] [Batch 525/938] [D loss: 0.240517] [G loss: 1.878493]
[Epoch 199/200] [Batch 526/938] [D loss: 0.300506] [G loss: 2.804384]
[Epoch 199/200] [Batch 527/938] [D loss: 0.197670] [G loss: 2.745258]
[Epoch 199/200] [Batch 528/938] [D loss: 0.294735] [G loss: 2.987031]
[Epoch 199/200] [Batch 529/938] [D loss: 0.255497] [G loss: 1.373394]
[Epoch 199/200] [Batch 530/938] [D loss: 0.286040] [G loss: 5.190227]
[Epoch 199/200] [Batch 531/938] [D loss: 0.259803] [G loss: 4.077757]
[Epoch 199/200] [Batch 532/938] [D loss: 0.304753] [G loss: 1.869082]
[Epoch 199/200] [Batch 533/938] [D loss: 0.207942] [G loss: 3.021339]
[Epoch 199/200] [Batch 534/938] [D loss: 0.264321] [G loss: 2.753015]
[Epoch 199/200] [Batch 535/938] [D loss: 0.370691] [G loss: 2.577611]
[Epoch 199/200] [Batch 536/938] [D loss: 0.212542] [G loss: 1.973422]
[Epoch 199/200] [Batch 537/938] [D loss: 0.332004] [G loss: 3.453413]
[Epoch 199/200] [Batch 538/938] [D loss: 0.261539] [G loss: 1.894632]
[Epoch 199/200] [Batch 539/938] [D loss: 0.354950] [G loss: 3.332237]
[Epoch 199/200] [Batch 540/938] [D loss: 0.213687] [G loss: 2.236458]
[Epoch 199/200] [Batch 541/938] [D loss: 0.239525] [G loss: 2.642477]
[Epoch 199/200] [Batch 542/938] [D loss: 0.341379] [G loss: 2.319110]
[Epoch 199/200] [Batch 543/938] [D loss: 0.322913] [G loss: 1.899036]
[Epoch 199/200] [Batch 544/938] [D loss: 0.481993] [G loss: 4.247181]
[Epoch 199/200] [Batch 545/938] [D loss: 0.205097] [G loss: 2.101673]
[Epoch 199/200] [Batch 546/938] [D loss: 0.184908] [G loss: 3.241096]
[Epoch 199/200] [Batch 547/938] [D loss: 0.233340] [G loss: 2.488641]
[Epoch 199/200] [Batch 548/938] [D loss: 0.164190] [G loss: 2.311397]
[Epoch 199/200] [Batch 549/938] [D loss: 0.352604] [G loss: 3.416824]
[Epoch 199/200] [Batch 550/938] [D loss: 0.285543] [G loss: 1.662718]
[Epoch 199/200] [Batch 551/938] [D loss: 0.400492] [G loss: 4.087183]
[Epoch 199/200] [Batch 552/938] [D loss: 0.213271] [G loss: 2.167628]
[Epoch 199/200] [Batch 553/938] [D loss: 0.199520] [G loss: 2.300820]
[Epoch 199/200] [Batch 554/938] [D loss: 0.247988] [G loss: 3.139075]
[Epoch 199/200] [Batch 555/938] [D loss: 0.365490] [G loss: 2.949702]
[Epoch 199/200] [Batch 556/938] [D loss: 0.398493] [G loss: 1.160478]
[Epoch 199/200] [Batch 557/938] [D loss: 0.401345] [G loss: 5.132848]
[Epoch 199/200] [Batch 558/938] [D loss: 0.371697] [G loss: 3.134183]
[Epoch 199/200] [Batch 559/938] [D loss: 0.488879] [G loss: 0.794484]
[Epoch 199/200] [Batch 560/938] [D loss: 0.407649] [G loss: 5.011094]
[Epoch 199/200] [Batch 561/938] [D loss: 0.212616] [G loss: 4.245358]
[Epoch 199/200] [Batch 562/938] [D loss: 0.400635] [G loss: 2.719585]
[Epoch 199/200] [Batch 563/938] [D loss: 0.552835] [G loss: 1.090494]
[Epoch 199/200] [Batch 564/938] [D loss: 0.283796] [G loss: 4.862352]
[Epoch 199/200] [Batch 565/938] [D loss: 0.295148] [G loss: 3.984716]
[Epoch 199/200] [Batch 566/938] [D loss: 0.282887] [G loss: 2.306898]
[Epoch 199/200] [Batch 567/938] [D loss: 0.187834] [G loss: 2.646323]
[Epoch 199/200] [Batch 568/938] [D loss: 0.295460] [G loss: 2.984094]
[Epoch 199/200] [Batch 569/938] [D loss: 0.260562] [G loss: 2.212486]
[Epoch 199/200] [Batch 570/938] [D loss: 0.178674] [G loss: 2.924784]
[Epoch 199/200] [Batch 571/938] [D loss: 0.138816] [G loss: 2.680943]
[Epoch 199/200] [Batch 572/938] [D loss: 0.220222] [G loss: 2.732858]
[Epoch 199/200] [Batch 573/938] [D loss: 0.164758] [G loss: 2.259133]
[Epoch 199/200] [Batch 574/938] [D loss: 0.159920] [G loss: 2.691353]
[Epoch 199/200] [Batch 575/938] [D loss: 0.225088] [G loss: 2.700349]
[Epoch 199/200] [Batch 576/938] [D loss: 0.203943] [G loss: 1.922646]
[Epoch 199/200] [Batch 577/938] [D loss: 0.184320] [G loss: 3.285455]
[Epoch 199/200] [Batch 578/938] [D loss: 0.155756] [G loss: 2.853138]
[Epoch 199/200] [Batch 579/938] [D loss: 0.194625] [G loss: 2.009051]
[Epoch 199/200] [Batch 580/938] [D loss: 0.185460] [G loss: 2.173333]
[Epoch 199/200] [Batch 581/938] [D loss: 0.148214] [G loss: 2.975968]
[Epoch 199/200] [Batch 582/938] [D loss: 0.221584] [G loss: 2.577449]
[Epoch 199/200] [Batch 583/938] [D loss: 0.227756] [G loss: 1.844349]
[Epoch 199/200] [Batch 584/938] [D loss: 0.177731] [G loss: 3.079261]
[Epoch 199/200] [Batch 585/938] [D loss: 0.150481] [G loss: 2.912713]
[Epoch 199/200] [Batch 586/938] [D loss: 0.242641] [G loss: 2.384207]
[Epoch 199/200] [Batch 587/938] [D loss: 0.237414] [G loss: 1.719103]
[Epoch 199/200] [Batch 588/938] [D loss: 0.149274] [G loss: 3.148543]
[Epoch 199/200] [Batch 589/938] [D loss: 0.237314] [G loss: 3.060877]
[Epoch 199/200] [Batch 590/938] [D loss: 0.234944] [G loss: 1.987834]
[Epoch 199/200] [Batch 591/938] [D loss: 0.152528] [G loss: 2.175102]
[Epoch 199/200] [Batch 592/938] [D loss: 0.237205] [G loss: 3.061235]
[Epoch 199/200] [Batch 593/938] [D loss: 0.222224] [G loss: 2.050287]
[Epoch 199/200] [Batch 594/938] [D loss: 0.187793] [G loss: 1.890448]
[Epoch 199/200] [Batch 595/938] [D loss: 0.199674] [G loss: 3.458822]
[Epoch 199/200] [Batch 596/938] [D loss: 0.152558] [G loss: 2.564784]
[Epoch 199/200] [Batch 597/938] [D loss: 0.142358] [G loss: 2.096752]
[Epoch 199/200] [Batch 598/938] [D loss: 0.275575] [G loss: 3.182770]
[Epoch 199/200] [Batch 599/938] [D loss: 0.324809] [G loss: 2.153387]
[Epoch 199/200] [Batch 600/938] [D loss: 0.259015] [G loss: 2.138117]
[Epoch 199/200] [Batch 601/938] [D loss: 0.217977] [G loss: 2.588317]
[Epoch 199/200] [Batch 602/938] [D loss: 0.078397] [G loss: 2.793597]
[Epoch 199/200] [Batch 603/938] [D loss: 0.185497] [G loss: 2.981017]
[Epoch 199/200] [Batch 604/938] [D loss: 0.204808] [G loss: 2.822205]
[Epoch 199/200] [Batch 605/938] [D loss: 0.144663] [G loss: 2.247668]
[Epoch 199/200] [Batch 606/938] [D loss: 0.059445] [G loss: 3.570878]
[Epoch 199/200] [Batch 607/938] [D loss: 0.309016] [G loss: 3.968372]
[Epoch 199/200] [Batch 608/938] [D loss: 0.211270] [G loss: 1.808640]
[Epoch 199/200] [Batch 609/938] [D loss: 0.208046] [G loss: 1.998029]
[Epoch 199/200] [Batch 610/938] [D loss: 0.165263] [G loss: 3.964797]
[Epoch 199/200] [Batch 611/938] [D loss: 0.201335] [G loss: 3.458240]
[Epoch 199/200] [Batch 612/938] [D loss: 0.167787] [G loss: 2.253948]
[Epoch 199/200] [Batch 613/938] [D loss: 0.141789] [G loss: 2.860279]
[Epoch 199/200] [Batch 614/938] [D loss: 0.185540] [G loss: 2.718572]
[Epoch 199/200] [Batch 615/938] [D loss: 0.162152] [G loss: 2.587972]
[Epoch 199/200] [Batch 616/938] [D loss: 0.123033] [G loss: 2.600150]
[Epoch 199/200] [Batch 617/938] [D loss: 0.143506] [G loss: 3.047388]
[Epoch 199/200] [Batch 618/938] [D loss: 0.228789] [G loss: 3.261727]
[Epoch 199/200] [Batch 619/938] [D loss: 0.216840] [G loss: 2.269515]
[Epoch 199/200] [Batch 620/938] [D loss: 0.183434] [G loss: 2.977046]
[Epoch 199/200] [Batch 621/938] [D loss: 0.229265] [G loss: 2.406548]
[Epoch 199/200] [Batch 622/938] [D loss: 0.071751] [G loss: 3.190362]
[Epoch 199/200] [Batch 623/938] [D loss: 0.160782] [G loss: 3.856777]
[Epoch 199/200] [Batch 624/938] [D loss: 0.191771] [G loss: 2.936582]
[Epoch 199/200] [Batch 625/938] [D loss: 0.217228] [G loss: 2.049066]
[Epoch 199/200] [Batch 626/938] [D loss: 0.097590] [G loss: 3.189340]
[Epoch 199/200] [Batch 627/938] [D loss: 0.216497] [G loss: 3.568003]
[Epoch 199/200] [Batch 628/938] [D loss: 0.211859] [G loss: 2.411761]
[Epoch 199/200] [Batch 629/938] [D loss: 0.218797] [G loss: 2.317430]
[Epoch 199/200] [Batch 630/938] [D loss: 0.218692] [G loss: 3.586989]
[Epoch 199/200] [Batch 631/938] [D loss: 0.151003] [G loss: 3.074996]
[Epoch 199/200] [Batch 632/938] [D loss: 0.175115] [G loss: 2.807910]
[Epoch 199/200] [Batch 633/938] [D loss: 0.308539] [G loss: 2.792548]
[Epoch 199/200] [Batch 634/938] [D loss: 0.203844] [G loss: 1.813756]
[Epoch 199/200] [Batch 635/938] [D loss: 0.196709] [G loss: 4.752981]
[Epoch 199/200] [Batch 636/938] [D loss: 0.486171] [G loss: 3.732586]
[Epoch 199/200] [Batch 637/938] [D loss: 0.384493] [G loss: 1.079856]
[Epoch 199/200] [Batch 638/938] [D loss: 0.309867] [G loss: 5.444859]
[Epoch 199/200] [Batch 639/938] [D loss: 0.158496] [G loss: 5.053759]
[Epoch 199/200] [Batch 640/938] [D loss: 0.166102] [G loss: 3.087509]
[Epoch 199/200] [Batch 641/938] [D loss: 0.269551] [G loss: 2.962107]
[Epoch 199/200] [Batch 642/938] [D loss: 0.160119] [G loss: 2.755856]
[Epoch 199/200] [Batch 643/938] [D loss: 0.203496] [G loss: 2.888652]
[Epoch 199/200] [Batch 644/938] [D loss: 0.259359] [G loss: 2.903952]
[Epoch 199/200] [Batch 645/938] [D loss: 0.221352] [G loss: 1.980564]
[Epoch 199/200] [Batch 646/938] [D loss: 0.188645] [G loss: 2.641405]
[Epoch 199/200] [Batch 647/938] [D loss: 0.137853] [G loss: 2.924153]
[Epoch 199/200] [Batch 648/938] [D loss: 0.159177] [G loss: 3.101475]
[Epoch 199/200] [Batch 649/938] [D loss: 0.222696] [G loss: 2.010757]
[Epoch 199/200] [Batch 650/938] [D loss: 0.221664] [G loss: 3.184425]
[Epoch 199/200] [Batch 651/938] [D loss: 0.298919] [G loss: 2.577543]
[Epoch 199/200] [Batch 652/938] [D loss: 0.173500] [G loss: 1.817697]
[Epoch 199/200] [Batch 653/938] [D loss: 0.148625] [G loss: 3.180742]
[Epoch 199/200] [Batch 654/938] [D loss: 0.365458] [G loss: 3.038286]
[Epoch 199/200] [Batch 655/938] [D loss: 0.202975] [G loss: 1.763512]
[Epoch 199/200] [Batch 656/938] [D loss: 0.392196] [G loss: 3.665395]
[Epoch 199/200] [Batch 657/938] [D loss: 0.161050] [G loss: 2.378026]
[Epoch 199/200] [Batch 658/938] [D loss: 0.187565] [G loss: 2.698604]
[Epoch 199/200] [Batch 659/938] [D loss: 0.151364] [G loss: 2.414559]
[Epoch 199/200] [Batch 660/938] [D loss: 0.236054] [G loss: 2.641492]
[Epoch 199/200] [Batch 661/938] [D loss: 0.164655] [G loss: 2.936527]
[Epoch 199/200] [Batch 662/938] [D loss: 0.172056] [G loss: 2.086460]
[Epoch 199/200] [Batch 663/938] [D loss: 0.195871] [G loss: 2.996851]
[Epoch 199/200] [Batch 664/938] [D loss: 0.275602] [G loss: 2.977741]
[Epoch 199/200] [Batch 665/938] [D loss: 0.437204] [G loss: 1.033299]
[Epoch 199/200] [Batch 666/938] [D loss: 0.323993] [G loss: 4.549184]
[Epoch 199/200] [Batch 667/938] [D loss: 0.273760] [G loss: 3.231051]
[Epoch 199/200] [Batch 668/938] [D loss: 0.342918] [G loss: 1.632046]
[Epoch 199/200] [Batch 669/938] [D loss: 0.280514] [G loss: 3.573195]
[Epoch 199/200] [Batch 670/938] [D loss: 0.219781] [G loss: 3.212118]
[Epoch 199/200] [Batch 671/938] [D loss: 0.314455] [G loss: 2.145201]
[Epoch 199/200] [Batch 672/938] [D loss: 0.375238] [G loss: 2.845634]
[Epoch 199/200] [Batch 673/938] [D loss: 0.373173] [G loss: 1.412919]
[Epoch 199/200] [Batch 674/938] [D loss: 0.313343] [G loss: 3.130870]
[Epoch 199/200] [Batch 675/938] [D loss: 0.228882] [G loss: 2.343262]
[Epoch 199/200] [Batch 676/938] [D loss: 0.195933] [G loss: 2.239018]
[Epoch 199/200] [Batch 677/938] [D loss: 0.273200] [G loss: 2.482838]
[Epoch 199/200] [Batch 678/938] [D loss: 0.198717] [G loss: 2.521555]
[Epoch 199/200] [Batch 679/938] [D loss: 0.175062] [G loss: 2.739739]
[Epoch 199/200] [Batch 680/938] [D loss: 0.203406] [G loss: 2.660036]
[Epoch 199/200] [Batch 681/938] [D loss: 0.166459] [G loss: 1.603495]
[Epoch 199/200] [Batch 682/938] [D loss: 0.348996] [G loss: 4.274703]
[Epoch 199/200] [Batch 683/938] [D loss: 0.169675] [G loss: 3.223457]
[Epoch 199/200] [Batch 684/938] [D loss: 0.253132] [G loss: 2.414246]
[Epoch 199/200] [Batch 685/938] [D loss: 0.304296] [G loss: 1.868444]
[Epoch 199/200] [Batch 686/938] [D loss: 0.184080] [G loss: 2.794784]
[Epoch 199/200] [Batch 687/938] [D loss: 0.155791] [G loss: 3.039576]
[Epoch 199/200] [Batch 688/938] [D loss: 0.176277] [G loss: 3.100944]
[Epoch 199/200] [Batch 689/938] [D loss: 0.132835] [G loss: 2.528110]
[Epoch 199/200] [Batch 690/938] [D loss: 0.205033] [G loss: 2.952233]
[Epoch 199/200] [Batch 691/938] [D loss: 0.233550] [G loss: 2.354641]
[Epoch 199/200] [Batch 692/938] [D loss: 0.202132] [G loss: 2.384284]
[Epoch 199/200] [Batch 693/938] [D loss: 0.212213] [G loss: 2.911900]
[Epoch 199/200] [Batch 694/938] [D loss: 0.198463] [G loss: 2.160588]
[Epoch 199/200] [Batch 695/938] [D loss: 0.213549] [G loss: 3.262148]
[Epoch 199/200] [Batch 696/938] [D loss: 0.120242] [G loss: 2.942937]
[Epoch 199/200] [Batch 697/938] [D loss: 0.185458] [G loss: 2.403639]
[Epoch 199/200] [Batch 698/938] [D loss: 0.227221] [G loss: 2.490418]
[Epoch 199/200] [Batch 699/938] [D loss: 0.170355] [G loss: 3.281399]
[Epoch 199/200] [Batch 700/938] [D loss: 0.222423] [G loss: 3.482199]
[Epoch 199/200] [Batch 701/938] [D loss: 0.156404] [G loss: 2.620489]
[Epoch 199/200] [Batch 702/938] [D loss: 0.111488] [G loss: 3.114704]
[Epoch 199/200] [Batch 703/938] [D loss: 0.118382] [G loss: 3.098218]
[Epoch 199/200] [Batch 704/938] [D loss: 0.116379] [G loss: 2.892557]
[Epoch 199/200] [Batch 705/938] [D loss: 0.149527] [G loss: 3.020519]
[Epoch 199/200] [Batch 706/938] [D loss: 0.167355] [G loss: 2.710308]
[Epoch 199/200] [Batch 707/938] [D loss: 0.159620] [G loss: 2.025631]
[Epoch 199/200] [Batch 708/938] [D loss: 0.130819] [G loss: 3.314302]
[Epoch 199/200] [Batch 709/938] [D loss: 0.175189] [G loss: 3.115243]
[Epoch 199/200] [Batch 710/938] [D loss: 0.174281] [G loss: 2.035674]
[Epoch 199/200] [Batch 711/938] [D loss: 0.135806] [G loss: 3.086106]
[Epoch 199/200] [Batch 712/938] [D loss: 0.108057] [G loss: 3.707244]
[Epoch 199/200] [Batch 713/938] [D loss: 0.139535] [G loss: 3.277218]
[Epoch 199/200] [Batch 714/938] [D loss: 0.154882] [G loss: 2.707584]
[Epoch 199/200] [Batch 715/938] [D loss: 0.200914] [G loss: 2.788248]
[Epoch 199/200] [Batch 716/938] [D loss: 0.104328] [G loss: 3.004518]
[Epoch 199/200] [Batch 717/938] [D loss: 0.106336] [G loss: 3.179773]
[Epoch 199/200] [Batch 718/938] [D loss: 0.258025] [G loss: 3.117836]
[Epoch 199/200] [Batch 719/938] [D loss: 0.194670] [G loss: 2.414116]
[Epoch 199/200] [Batch 720/938] [D loss: 0.233136] [G loss: 4.404694]
[Epoch 199/200] [Batch 721/938] [D loss: 0.220932] [G loss: 3.094989]
[Epoch 199/200] [Batch 722/938] [D loss: 0.189703] [G loss: 2.830696]
[Epoch 199/200] [Batch 723/938] [D loss: 0.255022] [G loss: 4.068260]
[Epoch 199/200] [Batch 724/938] [D loss: 0.249321] [G loss: 2.742521]
[Epoch 199/200] [Batch 725/938] [D loss: 0.319898] [G loss: 2.582678]
[Epoch 199/200] [Batch 726/938] [D loss: 0.369013] [G loss: 4.031104]
[Epoch 199/200] [Batch 727/938] [D loss: 0.266066] [G loss: 2.743596]
[Epoch 199/200] [Batch 728/938] [D loss: 0.201843] [G loss: 2.783618]
[Epoch 199/200] [Batch 729/938] [D loss: 0.367231] [G loss: 2.483276]
[Epoch 199/200] [Batch 730/938] [D loss: 0.349845] [G loss: 1.916077]
[Epoch 199/200] [Batch 731/938] [D loss: 0.352227] [G loss: 2.036606]
[Epoch 199/200] [Batch 732/938] [D loss: 0.409779] [G loss: 2.494383]
[Epoch 199/200] [Batch 733/938] [D loss: 0.380846] [G loss: 1.767339]
[Epoch 199/200] [Batch 734/938] [D loss: 0.400120] [G loss: 3.873574]
[Epoch 199/200] [Batch 735/938] [D loss: 0.327983] [G loss: 1.510524]
[Epoch 199/200] [Batch 736/938] [D loss: 0.211024] [G loss: 3.524242]
[Epoch 199/200] [Batch 737/938] [D loss: 0.202081] [G loss: 3.035087]
[Epoch 199/200] [Batch 738/938] [D loss: 0.185424] [G loss: 2.415739]
[Epoch 199/200] [Batch 739/938] [D loss: 0.214736] [G loss: 2.494403]
[Epoch 199/200] [Batch 740/938] [D loss: 0.269247] [G loss: 3.091796]
[Epoch 199/200] [Batch 741/938] [D loss: 0.386044] [G loss: 2.256771]
[Epoch 199/200] [Batch 742/938] [D loss: 0.343089] [G loss: 4.336098]
[Epoch 199/200] [Batch 743/938] [D loss: 0.243557] [G loss: 2.219161]
[Epoch 199/200] [Batch 744/938] [D loss: 0.222331] [G loss: 4.449853]
[Epoch 199/200] [Batch 745/938] [D loss: 0.222315] [G loss: 3.252462]
[Epoch 199/200] [Batch 746/938] [D loss: 0.240121] [G loss: 1.810207]
[Epoch 199/200] [Batch 747/938] [D loss: 0.193920] [G loss: 3.237525]
[Epoch 199/200] [Batch 748/938] [D loss: 0.326200] [G loss: 2.913218]
[Epoch 199/200] [Batch 749/938] [D loss: 0.224393] [G loss: 2.168216]
[Epoch 199/200] [Batch 750/938] [D loss: 0.168660] [G loss: 3.202345]
[Epoch 199/200] [Batch 751/938] [D loss: 0.213866] [G loss: 2.525848]
[Epoch 199/200] [Batch 752/938] [D loss: 0.234090] [G loss: 2.756093]
[Epoch 199/200] [Batch 753/938] [D loss: 0.181728] [G loss: 1.997999]
[Epoch 199/200] [Batch 754/938] [D loss: 0.243771] [G loss: 3.166817]
[Epoch 199/200] [Batch 755/938] [D loss: 0.295959] [G loss: 2.191718]
[Epoch 199/200] [Batch 756/938] [D loss: 0.257448] [G loss: 2.848547]
[Epoch 199/200] [Batch 757/938] [D loss: 0.174519] [G loss: 2.622894]
[Epoch 199/200] [Batch 758/938] [D loss: 0.275097] [G loss: 3.423331]
[Epoch 199/200] [Batch 759/938] [D loss: 0.306997] [G loss: 1.706573]
[Epoch 199/200] [Batch 760/938] [D loss: 0.267191] [G loss: 3.514220]
[Epoch 199/200] [Batch 761/938] [D loss: 0.252458] [G loss: 2.545773]
[Epoch 199/200] [Batch 762/938] [D loss: 0.269796] [G loss: 1.694119]
[Epoch 199/200] [Batch 763/938] [D loss: 0.294973] [G loss: 4.549989]
[Epoch 199/200] [Batch 764/938] [D loss: 0.252600] [G loss: 3.774599]
[Epoch 199/200] [Batch 765/938] [D loss: 0.199101] [G loss: 1.755407]
[Epoch 199/200] [Batch 766/938] [D loss: 0.175127] [G loss: 2.788262]
[Epoch 199/200] [Batch 767/938] [D loss: 0.254653] [G loss: 3.607895]
[Epoch 199/200] [Batch 768/938] [D loss: 0.191317] [G loss: 2.235488]
[Epoch 199/200] [Batch 769/938] [D loss: 0.168921] [G loss: 3.119192]
[Epoch 199/200] [Batch 770/938] [D loss: 0.205726] [G loss: 2.919157]
[Epoch 199/200] [Batch 771/938] [D loss: 0.272388] [G loss: 1.945572]
[Epoch 199/200] [Batch 772/938] [D loss: 0.186226] [G loss: 3.297346]
[Epoch 199/200] [Batch 773/938] [D loss: 0.135233] [G loss: 2.595504]
[Epoch 199/200] [Batch 774/938] [D loss: 0.179113] [G loss: 3.343285]
[Epoch 199/200] [Batch 775/938] [D loss: 0.210443] [G loss: 2.186341]
[Epoch 199/200] [Batch 776/938] [D loss: 0.136514] [G loss: 1.758584]
[Epoch 199/200] [Batch 777/938] [D loss: 0.134085] [G loss: 4.021963]
[Epoch 199/200] [Batch 778/938] [D loss: 0.174116] [G loss: 4.091988]
[Epoch 199/200] [Batch 779/938] [D loss: 0.300927] [G loss: 2.866177]
[Epoch 199/200] [Batch 780/938] [D loss: 0.333454] [G loss: 1.547460]
[Epoch 199/200] [Batch 781/938] [D loss: 0.231355] [G loss: 4.004800]
[Epoch 199/200] [Batch 782/938] [D loss: 0.274276] [G loss: 3.289218]
[Epoch 199/200] [Batch 783/938] [D loss: 0.234770] [G loss: 1.550313]
[Epoch 199/200] [Batch 784/938] [D loss: 0.183035] [G loss: 3.182945]
[Epoch 199/200] [Batch 785/938] [D loss: 0.250932] [G loss: 3.161758]
[Epoch 199/200] [Batch 786/938] [D loss: 0.199470] [G loss: 1.967526]
[Epoch 199/200] [Batch 787/938] [D loss: 0.155659] [G loss: 3.577609]
[Epoch 199/200] [Batch 788/938] [D loss: 0.176954] [G loss: 3.781487]
[Epoch 199/200] [Batch 789/938] [D loss: 0.138590] [G loss: 2.616993]
[Epoch 199/200] [Batch 790/938] [D loss: 0.152719] [G loss: 2.114896]
[Epoch 199/200] [Batch 791/938] [D loss: 0.129410] [G loss: 3.025338]
[Epoch 199/200] [Batch 792/938] [D loss: 0.156550] [G loss: 2.887554]
[Epoch 199/200] [Batch 793/938] [D loss: 0.232609] [G loss: 1.930881]
[Epoch 199/200] [Batch 794/938] [D loss: 0.108996] [G loss: 3.759365]
[Epoch 199/200] [Batch 795/938] [D loss: 0.160174] [G loss: 3.422358]
[Epoch 199/200] [Batch 796/938] [D loss: 0.205031] [G loss: 2.326794]
[Epoch 199/200] [Batch 797/938] [D loss: 0.248308] [G loss: 2.025723]
[Epoch 199/200] [Batch 798/938] [D loss: 0.187824] [G loss: 2.078479]
[Epoch 199/200] [Batch 799/938] [D loss: 0.130121] [G loss: 2.922575]
[Epoch 199/200] [Batch 800/938] [D loss: 0.179739] [G loss: 3.443364]
[Epoch 199/200] [Batch 801/938] [D loss: 0.147104] [G loss: 2.798279]
[Epoch 199/200] [Batch 802/938] [D loss: 0.160708] [G loss: 2.118432]
[Epoch 199/200] [Batch 803/938] [D loss: 0.077321] [G loss: 3.604582]
[Epoch 199/200] [Batch 804/938] [D loss: 0.335364] [G loss: 3.859048]
[Epoch 199/200] [Batch 805/938] [D loss: 0.197625] [G loss: 2.066278]
[Epoch 199/200] [Batch 806/938] [D loss: 0.196675] [G loss: 1.940624]
[Epoch 199/200] [Batch 807/938] [D loss: 0.095905] [G loss: 3.013686]
[Epoch 199/200] [Batch 808/938] [D loss: 0.298035] [G loss: 3.794077]
[Epoch 199/200] [Batch 809/938] [D loss: 0.226148] [G loss: 2.795662]
[Epoch 199/200] [Batch 810/938] [D loss: 0.241766] [G loss: 1.428153]
[Epoch 199/200] [Batch 811/938] [D loss: 0.198176] [G loss: 4.114351]
[Epoch 199/200] [Batch 812/938] [D loss: 0.221091] [G loss: 3.449719]
[Epoch 199/200] [Batch 813/938] [D loss: 0.133975] [G loss: 2.461325]
[Epoch 199/200] [Batch 814/938] [D loss: 0.136865] [G loss: 2.188529]
[Epoch 199/200] [Batch 815/938] [D loss: 0.205701] [G loss: 3.307138]
[Epoch 199/200] [Batch 816/938] [D loss: 0.278360] [G loss: 2.919518]
[Epoch 199/200] [Batch 817/938] [D loss: 0.241606] [G loss: 2.512475]
[Epoch 199/200] [Batch 818/938] [D loss: 0.214545] [G loss: 1.610157]
[Epoch 199/200] [Batch 819/938] [D loss: 0.210139] [G loss: 4.374883]
[Epoch 199/200] [Batch 820/938] [D loss: 0.274625] [G loss: 4.118075]
[Epoch 199/200] [Batch 821/938] [D loss: 0.266897] [G loss: 1.988297]
[Epoch 199/200] [Batch 822/938] [D loss: 0.219330] [G loss: 1.846941]
[Epoch 199/200] [Batch 823/938] [D loss: 0.225391] [G loss: 3.427641]
[Epoch 199/200] [Batch 824/938] [D loss: 0.178597] [G loss: 2.729839]
[Epoch 199/200] [Batch 825/938] [D loss: 0.251918] [G loss: 1.840112]
[Epoch 199/200] [Batch 826/938] [D loss: 0.108895] [G loss: 2.961050]
[Epoch 199/200] [Batch 827/938] [D loss: 0.359839] [G loss: 3.492851]
[Epoch 199/200] [Batch 828/938] [D loss: 0.359045] [G loss: 1.432290]
[Epoch 199/200] [Batch 829/938] [D loss: 0.276196] [G loss: 4.484563]
[Epoch 199/200] [Batch 830/938] [D loss: 0.217368] [G loss: 2.917485]
[Epoch 199/200] [Batch 831/938] [D loss: 0.231329] [G loss: 1.920922]
[Epoch 199/200] [Batch 832/938] [D loss: 0.192761] [G loss: 3.078996]
[Epoch 199/200] [Batch 833/938] [D loss: 0.172621] [G loss: 2.974698]
[Epoch 199/200] [Batch 834/938] [D loss: 0.261616] [G loss: 2.466723]
[Epoch 199/200] [Batch 835/938] [D loss: 0.151347] [G loss: 3.320613]
[Epoch 199/200] [Batch 836/938] [D loss: 0.238342] [G loss: 2.526172]
[Epoch 199/200] [Batch 837/938] [D loss: 0.384230] [G loss: 3.985128]
[Epoch 199/200] [Batch 838/938] [D loss: 0.294263] [G loss: 1.963407]
[Epoch 199/200] [Batch 839/938] [D loss: 0.229623] [G loss: 2.285572]
[Epoch 199/200] [Batch 840/938] [D loss: 0.184523] [G loss: 2.889511]
[Epoch 199/200] [Batch 841/938] [D loss: 0.263746] [G loss: 3.418724]
[Epoch 199/200] [Batch 842/938] [D loss: 0.299137] [G loss: 2.025144]
[Epoch 199/200] [Batch 843/938] [D loss: 0.352565] [G loss: 1.935957]
[Epoch 199/200] [Batch 844/938] [D loss: 0.275697] [G loss: 2.964591]
[Epoch 199/200] [Batch 845/938] [D loss: 0.351695] [G loss: 1.650733]
[Epoch 199/200] [Batch 846/938] [D loss: 0.324277] [G loss: 4.552853]
[Epoch 199/200] [Batch 847/938] [D loss: 0.281883] [G loss: 2.513546]
[Epoch 199/200] [Batch 848/938] [D loss: 0.400544] [G loss: 1.636672]
[Epoch 199/200] [Batch 849/938] [D loss: 0.276528] [G loss: 2.785306]
[Epoch 199/200] [Batch 850/938] [D loss: 0.132698] [G loss: 3.362943]
[Epoch 199/200] [Batch 851/938] [D loss: 0.299751] [G loss: 2.577307]
[Epoch 199/200] [Batch 852/938] [D loss: 0.215729] [G loss: 2.186105]
[Epoch 199/200] [Batch 853/938] [D loss: 0.190346] [G loss: 2.347000]
[Epoch 199/200] [Batch 854/938] [D loss: 0.241223] [G loss: 3.272163]
[Epoch 199/200] [Batch 855/938] [D loss: 0.197299] [G loss: 2.114619]
[Epoch 199/200] [Batch 856/938] [D loss: 0.160100] [G loss: 3.172706]
[Epoch 199/200] [Batch 857/938] [D loss: 0.156619] [G loss: 2.645803]
[Epoch 199/200] [Batch 858/938] [D loss: 0.170976] [G loss: 2.768240]
[Epoch 199/200] [Batch 859/938] [D loss: 0.278306] [G loss: 3.105035]
[Epoch 199/200] [Batch 860/938] [D loss: 0.276208] [G loss: 1.708549]
[Epoch 199/200] [Batch 861/938] [D loss: 0.305970] [G loss: 4.523860]
[Epoch 199/200] [Batch 862/938] [D loss: 0.261379] [G loss: 2.995561]
[Epoch 199/200] [Batch 863/938] [D loss: 0.213320] [G loss: 1.547480]
[Epoch 199/200] [Batch 864/938] [D loss: 0.206092] [G loss: 4.533790]
[Epoch 199/200] [Batch 865/938] [D loss: 0.201367] [G loss: 4.019772]
[Epoch 199/200] [Batch 866/938] [D loss: 0.341069] [G loss: 1.520725]
[Epoch 199/200] [Batch 867/938] [D loss: 0.203797] [G loss: 3.214973]
[Epoch 199/200] [Batch 868/938] [D loss: 0.177686] [G loss: 3.007144]
[Epoch 199/200] [Batch 869/938] [D loss: 0.205985] [G loss: 2.715195]
[Epoch 199/200] [Batch 870/938] [D loss: 0.146317] [G loss: 2.801755]
[Epoch 199/200] [Batch 871/938] [D loss: 0.202396] [G loss: 2.316250]
[Epoch 199/200] [Batch 872/938] [D loss: 0.219343] [G loss: 2.752495]
[Epoch 199/200] [Batch 873/938] [D loss: 0.189339] [G loss: 2.419459]
[Epoch 199/200] [Batch 874/938] [D loss: 0.232943] [G loss: 2.377493]
[Epoch 199/200] [Batch 875/938] [D loss: 0.125132] [G loss: 2.657328]
[Epoch 199/200] [Batch 876/938] [D loss: 0.190816] [G loss: 3.176470]
[Epoch 199/200] [Batch 877/938] [D loss: 0.151779] [G loss: 2.758420]
[Epoch 199/200] [Batch 878/938] [D loss: 0.157971] [G loss: 2.185449]
[Epoch 199/200] [Batch 879/938] [D loss: 0.200730] [G loss: 4.212247]
[Epoch 199/200] [Batch 880/938] [D loss: 0.230940] [G loss: 2.989897]
[Epoch 199/200] [Batch 881/938] [D loss: 0.292318] [G loss: 1.479537]
[Epoch 199/200] [Batch 882/938] [D loss: 0.216822] [G loss: 4.128265]
[Epoch 199/200] [Batch 883/938] [D loss: 0.189333] [G loss: 3.055676]
[Epoch 199/200] [Batch 884/938] [D loss: 0.220253] [G loss: 2.226689]
[Epoch 199/200] [Batch 885/938] [D loss: 0.150788] [G loss: 3.878600]
[Epoch 199/200] [Batch 886/938] [D loss: 0.182520] [G loss: 3.990966]
[Epoch 199/200] [Batch 887/938] [D loss: 0.294662] [G loss: 2.308296]
[Epoch 199/200] [Batch 888/938] [D loss: 0.105759] [G loss: 2.923222]
[Epoch 199/200] [Batch 889/938] [D loss: 0.249908] [G loss: 3.327215]
[Epoch 199/200] [Batch 890/938] [D loss: 0.224690] [G loss: 1.840214]
[Epoch 199/200] [Batch 891/938] [D loss: 0.203849] [G loss: 3.539769]
[Epoch 199/200] [Batch 892/938] [D loss: 0.187154] [G loss: 2.790998]
[Epoch 199/200] [Batch 893/938] [D loss: 0.164912] [G loss: 1.940909]
[Epoch 199/200] [Batch 894/938] [D loss: 0.291457] [G loss: 3.486979]
[Epoch 199/200] [Batch 895/938] [D loss: 0.279402] [G loss: 2.142149]
[Epoch 199/200] [Batch 896/938] [D loss: 0.372909] [G loss: 1.271861]
[Epoch 199/200] [Batch 897/938] [D loss: 0.290563] [G loss: 4.763008]
[Epoch 199/200] [Batch 898/938] [D loss: 0.156968] [G loss: 4.117105]
[Epoch 199/200] [Batch 899/938] [D loss: 0.222251] [G loss: 2.175460]
[Epoch 199/200] [Batch 900/938] [D loss: 0.101794] [G loss: 2.279027]
[Epoch 199/200] [Batch 901/938] [D loss: 0.242678] [G loss: 3.360278]
[Epoch 199/200] [Batch 902/938] [D loss: 0.222959] [G loss: 2.214505]
[Epoch 199/200] [Batch 903/938] [D loss: 0.238610] [G loss: 2.012178]
[Epoch 199/200] [Batch 904/938] [D loss: 0.332037] [G loss: 2.418947]
[Epoch 199/200] [Batch 905/938] [D loss: 0.233639] [G loss: 1.794948]
[Epoch 199/200] [Batch 906/938] [D loss: 0.230061] [G loss: 2.428849]
[Epoch 199/200] [Batch 907/938] [D loss: 0.118179] [G loss: 2.772367]
[Epoch 199/200] [Batch 908/938] [D loss: 0.140084] [G loss: 2.920207]
[Epoch 199/200] [Batch 909/938] [D loss: 0.231503] [G loss: 2.530897]
[Epoch 199/200] [Batch 910/938] [D loss: 0.168307] [G loss: 2.256913]
[Epoch 199/200] [Batch 911/938] [D loss: 0.184151] [G loss: 2.611748]
[Epoch 199/200] [Batch 912/938] [D loss: 0.138894] [G loss: 2.626979]
[Epoch 199/200] [Batch 913/938] [D loss: 0.239460] [G loss: 2.725363]
[Epoch 199/200] [Batch 914/938] [D loss: 0.180955] [G loss: 2.045964]
[Epoch 199/200] [Batch 915/938] [D loss: 0.265323] [G loss: 2.441764]
[Epoch 199/200] [Batch 916/938] [D loss: 0.318415] [G loss: 3.326228]
[Epoch 199/200] [Batch 917/938] [D loss: 0.231331] [G loss: 1.793122]
[Epoch 199/200] [Batch 918/938] [D loss: 0.144749] [G loss: 2.778381]
[Epoch 199/200] [Batch 919/938] [D loss: 0.156647] [G loss: 3.481919]
[Epoch 199/200] [Batch 920/938] [D loss: 0.260439] [G loss: 2.953190]
[Epoch 199/200] [Batch 921/938] [D loss: 0.298979] [G loss: 1.503320]
[Epoch 199/200] [Batch 922/938] [D loss: 0.372069] [G loss: 4.011266]
[Epoch 199/200] [Batch 923/938] [D loss: 0.198799] [G loss: 2.014938]
[Epoch 199/200] [Batch 924/938] [D loss: 0.213883] [G loss: 2.763435]
[Epoch 199/200] [Batch 925/938] [D loss: 0.238556] [G loss: 2.615246]
[Epoch 199/200] [Batch 926/938] [D loss: 0.217402] [G loss: 2.082436]
[Epoch 199/200] [Batch 927/938] [D loss: 0.253011] [G loss: 3.274029]
[Epoch 199/200] [Batch 928/938] [D loss: 0.262783] [G loss: 2.570278]
[Epoch 199/200] [Batch 929/938] [D loss: 0.249624] [G loss: 2.085838]
[Epoch 199/200] [Batch 930/938] [D loss: 0.118342] [G loss: 2.988968]
[Epoch 199/200] [Batch 931/938] [D loss: 0.333134] [G loss: 2.985602]
[Epoch 199/200] [Batch 932/938] [D loss: 0.331390] [G loss: 1.392959]
[Epoch 199/200] [Batch 933/938] [D loss: 0.299164] [G loss: 4.115664]
[Epoch 199/200] [Batch 934/938] [D loss: 0.275496] [G loss: 3.022765]
[Epoch 199/200] [Batch 935/938] [D loss: 0.436105] [G loss: 1.169448]
[Epoch 199/200] [Batch 936/938] [D loss: 0.335107] [G loss: 5.212241]
[Epoch 199/200] [Batch 937/938] [D loss: 0.164934] [G loss: 4.740316]
```









