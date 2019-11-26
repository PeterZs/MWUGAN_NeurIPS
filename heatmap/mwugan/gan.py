import tqdm
import os
import numpy as np
import time
from torch.nn import functional as F
import copy
import torch
import utils
from nnzoo import *

class GaussianSampler(object):
    def __init__(self, dim, for_img=False):
        self.dim = dim
        self.for_img = for_img

    def sampling(self, n):
        res = torch.randn(n, self.dim)
        if self.for_img:
            return res[:, :, None, None]
        else:
            return res


class UniformSampler(object):
    def __init__(self, dim):
        self.dim = dim

    def sampling(self, n):
        res = torch.randn(n, self.dim)
        if self.for_img:
            return res[:, :, None, None]
        else:
            return res


def loadNN(conf):
    if conf['name'] == 'dcgan_g':
        model = Generator_dcgan(dim_in=conf['z_dim'], hidden_size=conf['hidden_size'], channel=conf['channel'])
    elif conf['name'] == 'dcgan_g_ada':
        model = Generator_dcgan_adastyle(dim_in=conf['z_dim'], hidden_size=conf['hidden_size'], channel=conf['channel'])
    elif conf['name'] == 'dcgan_d':
        model = Discriminator_dcgan(hidden_size=conf['hidden_size'], channel=conf['channel'])
    elif conf['name'] == 'dcgan_d_ada':
        model = Discriminator_dcgan_adastyle(hidden_size=conf['hidden_size'], channel=conf['channel'])
    elif conf['name'] == 'gan_g':
        model = Generator_gan(input_size=conf['input_size'], hidden_size=conf['hidden_size'], output_size=conf['output_size'])
    elif conf['name'] == 'gan_d':
        model = Discriminator_gan(input_size=conf['input_size'], hidden_size=conf['hidden_size'], output_size=conf['output_size'])
    else:
        raise ValueError("no such NN name "+conf['name'])
    return model

class GAN(object):
    def __init__(self, gen_conf, dis_conf, sampler_type, dataset, batch_size, device):
        """
        """
        self.gen = loadNN(gen_conf)
        self.dis = loadNN(dis_conf)
        self.to_device(device)

        self.z_dim = gen_conf['z_dim']
        self.dataset = dataset
        if sampler_type == "gaussian":
            self.sampler = GaussianSampler(gen_conf['z_dim'], for_img=dataset.isimg)
        elif sampler_type == "uniform":
            self.sampler = UniformSampler(gen_conf['z_dim'], for_img=dataset.isimg)
        self.batch_size = batch_size

        self.g_opt = torch.optim.Adam(self.gen.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.d_opt = torch.optim.Adam(self.dis.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.device = device

        self.loss = nn.BCELoss()

        self.d_losses = []
        self.g_losses = []


    def to_device(self, device):
        self.gen.to(device=device)
        self.dis.to(device=device)

    def to_cpu(self):
        d = torch.device('cpu')
        self.gen.to(device=d)

    def train_mode(self):
        self.gen.train()
        self.dis.train()

    def eval_mode(self):
        self.gen.eval()
        self.dis.eval()

    def gen_iter(self):
        self.gen.zero_grad()
        latent_samples = self.sampler.sampling(self.batch_size)
        g_gen_input = latent_samples.to(dtype=torch.float32, device=self.device)
        g_fake_data = self.gen(g_gen_input)
        g_fake_deision = self.dis(g_fake_data)
        g_fake_labels = torch.ones(g_fake_deision.shape, dtype=torch.float32, device=self.device)

        g_loss = self.loss(g_fake_deision, g_fake_labels)
        self.g_losses.append(g_loss.item())
        g_loss.backward()
        self.g_opt.step()

    def dis_iter(self):
        self.dis.zero_grad()
        real_samples, _ = self.dataset.sample_batch(self.batch_size)
        if isinstance(real_samples, list):
            real_samples = real_samples[0]
        d_real_data = real_samples.to(dtype=torch.float32, device=self.device)
        d_real_decision = self.dis(d_real_data)
        d_real_labels = torch.ones(d_real_decision.shape, dtype=torch.float32, device=self.device)
        d_real_loss = self.loss(d_real_decision, d_real_labels)


        latent_samples = self.sampler.sampling(self.batch_size)
        d_gen_input = latent_samples.to(dtype=torch.float32, device=self.device)
        d_fake_data = self.gen(d_gen_input)
        d_fake_decision = self.dis(d_fake_data)
        d_fake_labels = torch.zeros(d_fake_decision.shape, dtype=torch.float32, device=self.device)
        d_fake_loss = self.loss(d_fake_decision, d_fake_labels)


        d_loss = d_real_loss + d_fake_loss
        self.d_losses.append(d_loss.item())
        d_loss.backward()
        self.d_opt.step()


    def train(self, iters, gen_step=1, dis_step=1):
        for _ in tqdm.tqdm(range(iters)):
            for gi in range(gen_step):
                self.dis_iter()
            for di in range(gen_step):
                self.gen_iter()

    def train_classifier(self, nn_conf, iters, model=None, augment=False):
        if model is None:
            if nn_conf['name'] == 'dcgan_d':
                model = Discriminator_dcgan(hidden_size=nn_conf['hidden_size'], channel=nn_conf['channel'])
            else:
                raise NotImplementedError
        pbar = tqdm.tqdm(range(iters))
        model.to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
        losses = []
        for i in pbar:
            model.zero_grad()

            real_samples, _ = self.dataset.sample_batch(self.batch_size)
            if augment:
                real_samples = utils.random_noise(real_samples, 0.1)
            d_real_data = real_samples.to(dtype=torch.float32, device=self.device)
            d_real_decision = model(d_real_data)
            d_real_labels = torch.ones(d_real_decision.shape, dtype=torch.float32, device=self.device)
            d_real_loss = self.loss(d_real_decision, d_real_labels)

            latent_samples = self.sampler.sampling(self.batch_size)
            d_gen_input = latent_samples.to(dtype=torch.float32, device=self.device)
            d_fake_data = self.gen(d_gen_input)
            if augment:
                d_fake_data = d_fake_data.to(device=torch.device('cpu'))
                d_fake_data = utils.random_noise(d_fake_data, 0.1)
                d_fake_data = d_fake_data.to(dtype=torch.float32, device=self.device)
            d_fake_decision = model(d_fake_data)
            d_fake_labels = torch.zeros(d_fake_decision.shape, dtype=torch.float32, device=self.device)
            d_fake_loss = self.loss(d_fake_decision, d_fake_labels)

            total_loss = d_real_loss + d_fake_loss
            losses.append(total_loss.item())
            total_loss.backward()
            opt.step()
        return model, losses


    def gen_samples(self, n_samples):
        latent_samples = self.sampler.sampling(n_samples)
        d_gen_input = latent_samples.to(dtype=torch.float32, device=self.device)
        output = self.gen(d_gen_input)
        return output

    def eval_fake(self, n_samples):
        latent_samples = self.sampler.sampling(n_samples)
        d_gen_input = latent_samples.to(dtype=torch.float32, device=self.device)
        d_fake_data = self.gen(d_gen_input)
        d_fake_decision = self.dis(d_fake_data)
        return d_fake_data, d_fake_decision


class MWUGAN(object):
    def __init__(self, config, dataset, device, gan_batchsize, update_method='mwu', delta=0.25):
        self.gan_list = []
        self.weights_list = []
        self.dis_list = []
        self.conf = config
        self.dataset = dataset
        self.device = device
        self.batch_size = gan_batchsize
        self.update_method = update_method
        assert(update_method in ['mwu', 'naive'])
        self.delta = delta

        self.g_step = 1
        self.d_step = 1

    
    def train_single(self, gan_iters, dis_iters, use_old=False):
        if use_old:
            if(len(self.gan_list)-len(self.weights_list) != 1):
                raise ValueError('should use a new GAN after weights update')
            gan = self.gan_list[-1]
            if(len(self.weights_list)==0):
                gan.dataset.sample_weights = np.ones(len(gan.dataset))
            else:
                gan.dataset.sample_weights = self.weights_list[-1]
        else:
            gan = GAN(gen_conf=self.conf['gen_conf'], dis_conf=self.conf['dis_conf'], sampler_type=self.conf['sampler'], dataset=self.dataset, batch_size=self.batch_size, device=self.device)
            if(len(self.weights_list)==0):
                gan.dataset.sample_weights = np.ones(len(gan.dataset))
            else:
                gan.dataset.sample_weights = self.weights_list[-1]
        gan.train(iters=dis_iters, gen_step=self.g_step, dis_step=self.d_step)
        self.dis_list.append(gan.dis)
        gan.train(iters=gan_iters, gen_step=self.g_step, dis_step=self.d_step)
        if not use_old:
            self.gan_list.append(gan)

    def eval_conf_d(self):
        confidence = np.zeros(len(self.dataset))
        for i in range(len(self.dataset)):
            img, label = self.dataset[i:i+1]
            d_real_data = img.to(dtype=torch.float32, device=self.device)
            c = self.dis_list[-1](d_real_data)
            confidence[i] = c.cpu().detach()
        return confidence

    def iteration(self, gan_iters, dis_iters):
        self.train_single(gan_iters=gan_iters, dis_iters=dis_iters)
        confidence = self.eval_conf_d()
        if self.update_method == 'mwu':
            p = self.mwu(confidence)
        elif self.update_method == 'naive':
            p = self.mwu_naive(confidence)
        self.weights_list.append(p)
        assert(len(self.gan_list)==len(self.weights_list))


    def mwu_naive(self, confidence):
        p = np.ones(len(confidence))
        return p

    def mwu(self, confidence):
        p = np.zeros(len(confidence))
        n = len(p)
        if(len(self.weights_list)==0):
            last_sum = n
        else:
            last_sum = np.sum(self.weights_list[-1])
        for i in range(n):
            c_conf = confidence[i]
            a = 1/(c_conf+1e-8)-1

            if(len(self.weights_list)==0):
                last_p = 1
            else:
                last_p = self.weights_list[-1][i]
            
            p0 = last_p/last_sum
            term = a*p0*n
            if(term<self.delta):
                p[i] = last_p*2
            else:
                p[i] = last_p
            
        return p
