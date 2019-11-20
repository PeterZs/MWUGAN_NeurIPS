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
        # a = (1/(confidence+1e-8)-1)
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

'''
class EnsembleGAN(object):
    def __init__(self, config, log_path, epochs, gan_epochs, gan_config, device,
                 classifier_cfg=None):
        self.classifier_cfg = classifier_cfg
        self.config = config
        self.log_path = log_path
        self.epochs = epochs
        self.gan_epochs = gan_epochs
        self.gan_config = gan_config
        self.device = device

        self.loader_config = gan_config.loader_config
        self.dis_config = gan_config.dis_config
        self.gen_config = gan_config.gen_config

        self.gan_list = []


class MWUGAN(object):
    def __init__(self, config, log_path, epochs, gan_epochs, gan_config, device,
                 classifier_cfg=None, mwu_scale=2, same_init=False, mixture_classifier_cfg=None):
        self.classifier_cfg = classifier_cfg
        self.config = config
        self.log_path = log_path
        self.epochs = epochs
        self.gan_epochs = gan_epochs
        self.gan_config = gan_config
        self.device = device
        self.loader_config = gan_config.loader_config
        self.dis_config = gan_config.dis_config
        self.gen_config = gan_config.gen_config
        self.mwu_scale = mwu_scale
        self.same_init = same_init
        self.mixture_classifier_cfg = mixture_classifier_cfg
        self.gan_list = []

    def weights_update(self, gan, weights, device, truncate_threshold, mwu_scale):
        data = gan.dataset.data
        pdata = weights
        ## ? TODO
        # pdata = pdata / np.sum(pdata)

        if self.mixture_classifier_cfg is None:
            confidence = []
            for idx in range(data.shape[0]):
                d = torch.Tensor(np.expand_dims(data[idx], 0)).to(device)
                lh = gan.dis(d).cpu().detach().numpy().flatten()
                confidence.append(lh[0] if lh[0] > truncate_threshold else truncate_threshold)
            confidence = np.array(confidence)
        else:
            real_data = data
            # fake_data = self.get_gens_outputs(self.gan_list, len(real_data) // 100, self.gan_list[0].sampler, 100, self.device)
            fake_data = self.get_gens_outputs([gan], len(real_data) // 100, self.gan_list[0].sampler, 100, self.device)
            confidence = self.mixture_classifier_cfg.evaluate_real_data(real_data, fake_data)
            confidence[confidence < truncate_threshold] = truncate_threshold

        confidence[confidence < truncate_threshold] = truncate_threshold

        pgen = pdata / confidence - pdata
        if np.sum(pgen) > 0.0:
            pgen = pgen * pgen.shape[0] / np.sum(pgen)  # sum(pgen) = n

        weights[pgen.flatten() > 1. / 4] /= mwu_scale
        return weights, confidence

    @staticmethod
    def get_gens_outputs(list_of_gans, iterations, sam, batch_size,device):
        """
        :param iterations:
        :param sam: noise sampler
        :param device:
        :return:
        """
        generated_pts = []
        for i in range(iterations):
            gan = random.choice(list_of_gans)
            samples = sam.sampling(batch_size).to(device).float()
            gan.eval_mode()
            res = gan.gen(samples)
            generated_pts.append(res.cpu().detach().numpy())
        generated_pts = np.concatenate(generated_pts)
        return generated_pts

    @staticmethod
    def eval_with_classifier_data(classifier, data, classifier_threshold, device, along_width=True):
        classifier.eval()
        classifier.to(device)

        result_is_confident = []
        result = []
        result_probs = []

        inp1, inp2, inp3 = np.split(data, 3, axis=3 if along_width else 1)
        res1 = classifier(torch.Tensor(inp1).to(device)).cpu().detach().numpy()
        res2 = classifier(torch.Tensor(inp2).to(device)).cpu().detach().numpy()
        res3 = classifier(torch.Tensor(inp3).to(device)).cpu().detach().numpy()

        prob1 = np.exp(res1)
        prob2 = np.exp(res2)
        prob3 = np.exp(res3)

        prob1 = np.max(prob1, axis=1)
        prob2 = np.max(prob2, axis=1)
        prob3 = np.max(prob3, axis=1)

        res1 = np.argmax(res1, axis=1)
        res2 = np.argmax(res2, axis=1)
        res3 = np.argmax(res3, axis=1)

        res = res1 * 100 + res2 * 10 + res3
        result.append(res)
        result_is_confident.append(
            (prob1 > classifier_threshold) *
            (prob2 > classifier_threshold) *
            (prob3 > classifier_threshold)
        )
        result_probs.append(np.column_stack((prob1, prob2, prob3)))

        result = np.hstack(result)
        result_probs = np.vstack(result_probs)
        result_is_confident = np.hstack(result_is_confident)

        digits = result.astype(int)
        logging.debug(
            'Ratio of confident predictions: %.4f' % \
            np.mean(result_is_confident))
        # Plot one fake image per detected mode

        # Confidence of made predictions
        conf = np.mean(result_probs)
        if np.sum(result_is_confident) == 0:
            C_actual = 0.
            C = 0.
            JS = 2.
        else:
            # Compute the actual coverage
            C_actual = len(np.unique(digits[result_is_confident])) / 1000.
            # Compute the JS with uniform
            JS = js_div_uniform(digits)
            # Compute Pdata(Pmodel > t) where Pmodel( Pmodel > t ) = 0.95
            # np.percentaile(a, 10) returns t s.t. np.mean( a <= t ) = 0.1
            phat = np.bincount(digits[result_is_confident], minlength=1000)
            phat = (phat + 0.) / np.sum(phat)
            threshold = np.percentile(phat, 5)
            ratio_not_covered = np.mean(phat <= threshold)
            C = 1. - ratio_not_covered

        logging.info('Evaluating: JS=%.3f, C=%.3f, C_actual=%.3f, Confidence=%.4f\n' % (JS, C, C_actual, conf))
        return JS, C, C_actual, conf, phat

    def eval_with_classifier(self, t, classifier, classifier_threshold, along_width=True, high_quality_samples=True):
        classifier.eval()
        classifier.to(self.device)

        result_is_confident = []
        result = []
        result_probs = []

        if not high_quality_samples:
            eval_epochs = 1
            bs = 100
            eval_iters = 1000
            for _ in tqdm.tqdm(range(eval_epochs)):  # total num 1 * 128 * 100 = 12800
                cur_gan_output = self.get_gens_outputs(self.gan_list, eval_iters, self.gan_list[0].sampler, bs, self.device)
                for idx in range(cur_gan_output.shape[0] // bs):
                    inp1, inp2, inp3 = np.split(cur_gan_output[idx * bs: idx * bs + bs], 3, axis=3 if along_width else 1)
                    res1 = classifier(torch.Tensor(inp1).to(self.device)).cpu().detach().numpy()
                    res2 = classifier(torch.Tensor(inp2).to(self.device)).cpu().detach().numpy()
                    res3 = classifier(torch.Tensor(inp3).to(self.device)).cpu().detach().numpy()

                    prob1 = np.exp(res1)
                    prob2 = np.exp(res2)
                    prob3 = np.exp(res3)

                    prob1 = np.max(prob1, axis=1)
                    prob2 = np.max(prob2, axis=1)
                    prob3 = np.max(prob3, axis=1)

                    res1 = np.argmax(res1, axis=1)
                    res2 = np.argmax(res2, axis=1)
                    res3 = np.argmax(res3, axis=1)

                    res = res1 * 100 + res2 * 10 + res3

                    result.append(res)
                    result_is_confident.append(
                        (prob1 > classifier_threshold) *
                        (prob2 > classifier_threshold) *
                        (prob3 > classifier_threshold)
                    )
                    result_probs.append(np.column_stack((prob1, prob2, prob3)))

            np.save(os.patorch.join(self.log_path, 'result_{}.npy'.format(t)), cur_gan_output)

            result = np.hstack(result)
            result_probs = np.vstack(result_probs)
            result_is_confident = np.hstack(result_is_confident)

            digits = result.astype(int)
            logging.debug(
                'Ratio of confident predictions: %.4f' % \
                np.mean(result_is_confident))
            # Plot one fake image per detected mode
            # gathered = []
            # for (idx, dig) in enumerate(list(digits)):
            #     if not dig in gathered and result_is_confident[idx]:
            #         gathered.append(dig)
            #         p = result_probs[idx]
            #         logging.debug('Mode %03d covered with prob %.3f, %.3f, %.3f' % \
            #                       (dig, p[0], p[1], p[2]))
            # Confidence of made predictions
            conf = np.mean(result_probs)
            if np.sum(result_is_confident) == 0:
                C_actual = 0.
                C = 0.
                JS = 2.
            else:
                # Compute the actual coverage
                C_actual = len(np.unique(digits[result_is_confident])) / 1000.
                # Compute the JS with uniform
                JS = js_div_uniform(digits)
                # Compute Pdata(Pmodel > t) where Pmodel( Pmodel > t ) = 0.95
                # np.percentaile(a, 10) returns t s.t. np.mean( a <= t ) = 0.1
                phat = np.bincount(digits[result_is_confident], minlength=1000)
                phat = (phat + 0.) / np.sum(phat)
                threshold = np.percentile(phat, 5)
                ratio_not_covered = np.mean(phat <= threshold)
                C = 1. - ratio_not_covered

            logging.info('Evaluating: JS=%.3f, C=%.3f, C_actual=%.3f, Confidence=%.4f\n' % (JS, C, C_actual, conf))
            return JS, C, C_actual, conf
        else:
            n = 0
            bs = 100
            eval_iters = 10
            iterations = 0
            result_img = []
            while n < 100000 and iterations < 1000:
                cur_gan_output = self.get_gens_outputs(self.gan_list, eval_iters, self.gan_list[0].sampler, bs,
                                                       self.device)
                for idx in range(cur_gan_output.shape[0] // bs):
                    inp1, inp2, inp3 = np.split(cur_gan_output[idx * bs: idx * bs + bs], 3,
                                                axis=3 if along_width else 1)
                    res1 = classifier(torch.Tensor(inp1).to(self.device)).cpu().detach().numpy()
                    res2 = classifier(torch.Tensor(inp2).to(self.device)).cpu().detach().numpy()
                    res3 = classifier(torch.Tensor(inp3).to(self.device)).cpu().detach().numpy()

                    prob1 = np.exp(res1)
                    prob2 = np.exp(res2)
                    prob3 = np.exp(res3)

                    prob1 = np.max(prob1, axis=1)
                    prob2 = np.max(prob2, axis=1)
                    prob3 = np.max(prob3, axis=1)

                    res1 = np.argmax(res1, axis=1)
                    res2 = np.argmax(res2, axis=1)
                    res3 = np.argmax(res3, axis=1)

                    res = res1 * 100 + res2 * 10 + res3
                    res_is_confident = (prob1 > classifier_threshold) * (prob2 > classifier_threshold) * (prob3 > classifier_threshold)
                    result.append(res[res_is_confident])
                    result_img.append(cur_gan_output[idx * bs: idx * bs + bs][res_is_confident])
                    result_probs.append(
                        np.column_stack((prob1[res_is_confident], prob2[res_is_confident], prob3[res_is_confident]))
                    )
                    n += np.sum(res_is_confident)
                iterations += 1

            result_img = np.vstack(result_img)
            print(result_img.shape)
            result_probs = np.vstack(result_probs)
            result = np.hstack(result)
            digits = result.astype(int)

            # Compute the actual coverage
            C_actual = len(np.unique(digits[result_is_confident])) / 1000.
            # Compute the JS with uniform
            JS = js_div_uniform(digits)
            # Compute Pdata(Pmodel > t) where Pmodel( Pmodel > t ) = 0.95
            # np.percentaile(a, 10) returns t s.t. np.mean( a <= t ) = 0.1
            phat = np.bincount(digits[result_is_confident], minlength=1000)
            phat = (phat + 0.) / np.sum(phat)
            threshold = np.percentile(phat, 5)
            ratio_not_covered = np.mean(phat <= threshold)
            C = 1. - ratio_not_covered

            conf = np.mean(result_probs)
            np.save(os.patorch.join(self.log_path, 'result_label_{}.npy'.format(t)), result)
            np.save(os.patorch.join(self.log_path, 'result_{}.npy'.format(t)), result_img)
            logging.info('Evaluating: JS=%.3f, C=%.3f, C_actual=%.3f, Confidence=%.4f\n' % (JS, C, C_actual, conf))
            print('Evaluating: JS=%.3f, C=%.3f, C_actual=%.3f, Confidence=%.4f\n' % (JS, C, C_actual, conf))
            return JS, C, C_actual, conf

    def eval_with_classifier_simple(self, t, classifier, classifier_threshold):
        classifier.eval()
        classifier.to(self.device)

        result_is_confident = []
        result = []
        result_probs = []

        n = 0
        bs = 100
        eval_iters = 10
        iterations = 0
        result_img = []
        while n < 100000 and iterations < 1000:
            cur_gan_output = self.get_gens_outputs(self.gan_list, eval_iters, self.gan_list[0].sampler, bs,
                                                    self.device)
            for idx in range(cur_gan_output.shape[0] // bs):
                inp1 = cur_gan_output[idx * bs: idx * bs + bs]
                res1 = classifier(torch.Tensor(inp1).to(self.device)).cpu().detach().numpy()
                prob1 = np.exp(res1)
                prob1 = np.max(prob1, axis=1)
                res1 = np.argmax(res1, axis=1)
                res = res1
                res_is_confident = (prob1 > classifier_threshold)
                result.append(res[res_is_confident])
                result_img.append(cur_gan_output[idx * bs: idx * bs + bs][res_is_confident])
                result_probs.append(
                    np.column_stack((prob1[res_is_confident]))
                )
                n += np.sum(res_is_confident)
            iterations += 1

        result_img = np.vstack(result_img)
        print(result_img.shape)
        result_probs = np.hstack(result_probs).T
        result = np.hstack(result)
        digits = result.astype(int)

        # Compute the actual coverage
        C_actual = len(np.unique(digits[result_is_confident])) / 1000.
        # Compute the JS with uniform
        JS = js_div_uniform(digits)
        # Compute Pdata(Pmodel > t) where Pmodel( Pmodel > t ) = 0.95
        # np.percentaile(a, 10) returns t s.t. np.mean( a <= t ) = 0.1
        phat = np.bincount(digits[result_is_confident], minlength=1000)
        phat = (phat + 0.) / np.sum(phat)
        threshold = np.percentile(phat, 5)
        ratio_not_covered = np.mean(phat <= threshold)
        C = 1. - ratio_not_covered

        conf = np.mean(result_probs)
        np.save(os.patorch.join(self.log_path, 'result_label_{}.npy'.format(t)), result)
        np.save(os.patorch.join(self.log_path, 'result_{}.npy'.format(t)), result_img)
        logging.info('Evaluating: JS=%.3f, C=%.3f, C_actual=%.3f, Confidence=%.4f\n' % (JS, C, C_actual, conf))
        print('Evaluating: JS=%.3f, C=%.3f, C_actual=%.3f, Confidence=%.4f\n' % (JS, C, C_actual, conf))
        return JS, C, C_actual, conf

    def train_with_classifier2(self, epochs=None, gan_epochs=None, classifier_thre=0.95):
        if epochs is None:
            epochs = self.epochs

        if gan_epochs is None:
            epochs = self.gan_epochs

        for t in range(epochs):
            cur_gan = self.gan_config.get()
            self.train_single_epoch(cur_gan, gan_epochs=gan_epochs)
            self.gan_list.append(cur_gan)
            classifier = self.train_classifier() 
            self.weigth_update2(classifier=classifier, classifier_thre=classifier_thre)
    
    def train_classifier(self):
        pass

    def train_single_epoch(self, gan, gan_epochs=None):
        gan.train_mode()
        if gan_epochs is None:
            gan_epochs = self.gan_epochs
        gan.train(gan_epochs)
        pass

    def weigth_update2(self, classifier, classifier_thre):
        pass
        

    def train_with_classifier(self, epochs=None, gan_epochs=None, classifier_threshold=0.95, along_width=False, high_quality_samples=True):
        print('*' * 5 + 'start to train MWUGAN' + '*' * 5)
        logging.info('*' * 5 + 'start to train MWUGAN' + '*' * 5)

        classifier = self.classifier_cfg.get()
        cur_gan = None

        gen_state_dict = None
        dis_state_dict = None
        if epochs is None:
            epochs = self.epochs

        if gan_epochs is None:
            epochs = self.gan_epochs

        for t in range(epochs):
            print('Train # {}/{}'.format(t, self.epochs))
            if cur_gan is None:
                cur_gan = self.gan_config.get()
                if self.same_init:
                    gen_state_dict = copy.deepcopy(cur_gan.gen.state_dict())
                    dis_state_dict = copy.deepcopy(cur_gan.dis.state_dict())
            else:
                cur_gan = self.gan_config.get(cur_gan.dataloader.dataset)
                if self.same_init:
                    cur_gan.gen.load_state_dict(gen_state_dict)
                    cur_gan.dis.load_state_dict(dis_state_dict)
            weights = cur_gan.dataloader.sampler.weights.numpy()
            cur_gan.train_mode()
            cur_gan.train(gan_epochs)

            self.gan_list.append(cur_gan)

            # do weights update
            print('Weights updating... ', end='')
            weights, confidence = self.weights_update(cur_gan, weights, self.device, 0.1, self.mwu_scale)
            self.loader_config.weights = weights
            self.eval_with_classifier_simple(t, classifier, classifier_threshold)
            # self.eval_with_classifier(t, classifier, classifier_threshold, along_width, high_quality_samples)

            np.savez(os.patorch.join(self.log_path, 'weight_step{}.npy'.format(t)), weights)
            np.savez(os.patorch.join(self.log_path, 'conf_step{}.npy'.format(t)), confidence)
    
    def train(self, eval=False, vis=False, save_samples=True, iters=0):
        print('*' * 5 + 'start to train MWUGAN' + '*' * 5)
        self.gan_list = []
        cur_gan = None
        for t in range(self.epochs):
            print('Train # {}/{}'.format(t, self.epochs))
            if cur_gan is None:
                cur_gan = self.gan_config.get()
                if self.same_init:
                    gen_state_dict = copy.deepcopy(cur_gan.gen.state_dict())
                    dis_state_dict = copy.deepcopy(cur_gan.dis.state_dict())
            else:
                cur_gan = self.gan_config.get(cur_gan.dataloader.dataset)
                if self.same_init:
                    cur_gan.gen.load_state_dict(gen_state_dict)
                    cur_gan.dis.load_state_dict(dis_state_dict)

            weights = cur_gan.dataloader.sampler.weights.numpy()
            cur_gan.train(self.gan_epochs)

            self.gan_list.append(cur_gan)

            # do weights update
            print('Weights updating... ', end='')
            weights, confidence = self.weights_update(cur_gan, weights, self.device, 0.1, self.mwu_scale)
            self.loader_config.weights = weights
            data = cur_gan.dataset.data
            xrange = [[np.min(data[:, 0]), np.max(data[:, 0])], [np.min(data[:, 1]), np.max(data[:, 1])]]

            if eval:
                # evaluation
                print('Evaluation...', end='')
                cur_gan_output = self.get_gens_outputs([cur_gan], 10, cur_gan.sampler, 100, self.device)
                all_gans_output = self.get_gens_outputs(self.gan_list, 500, cur_gan.sampler, 100, self.device)
                more_gans_output = self.get_gens_outputs(self.gan_list, 10, cur_gan.sampler, 100, self.device)
                if save_samples:
                    np.savez(os.patorch.join(self.log_path, 'only_iter{}_step{}.npy'.format(iters, t)), cur_gan_output)
                    np.savez(os.patorch.join(self.log_path, 'iter{}_step{}.npy'.format(iters, t)), all_gans_output)
                np.savez(os.patorch.join(self.log_path, 'weight_iter{}_step{}.npy'.format(iters, t)), weights)
                np.savez(os.patorch.join(self.log_path, 'conf_iter{}_step{}.npy'.format(iters, t)), confidence)
                random_data = data[np.random.choice(data.shape[0], 1000), :]
                likelihood, coverage = synthetic_coverage(all_gans_output, random_data, more_gans_output)
                logging.info('Epoch {}, likelihood {}, coverage {}'.format(t, likelihood, coverage))
                print('Epoch {}, likelihood {}, coverage {}'.format(t, likelihood, coverage))

                # save L and C to file
                with open(os.patorch.join(self.log_path, 'result.txt'), 'a') as f:
                    f.write('{},{},{}\n'.format(t, likelihood, coverage))

            if vis:
                # visualization
                print('Visualizing...')
                if t == 0:
                    visualize.heatmap(
                        os.patorch.join(self.log_path, 'target_dist.jpg'),
                        data,
                        'target',
                        xrange=xrange
                    )

                visualize.heatmap(
                    os.patorch.join(self.log_path, 'cur_gens_heatmap_{:03d}.jpg'.format(t)),
                    cur_gan_output,
                    'cur generator output epoch {}'.format(t),
                    xrange=xrange
                )
                visualize.discriminator_heatmap(
                    os.patorch.join(self.log_path, 'dis_confi_{:03d}.jpg'.format(t)),
                    'discriminator confidence epoch {}'.format(t),
                    cur_gan.dis,
                    xrange=xrange,
                    device=self.device
                )

                visualize.heatmap(
                    os.patorch.join(self.log_path, 'all_gens_heatmap_{:03d}.jpg'.format(t)),
                    all_gans_output,
                    'all generators output epoch {}'.format(t),
                    xrange=xrange
                )
                visualize.show_weight(
                    os.patorch.join(self.log_path, 'weights_{:02d}.jpg'.format(t)),
                    weights,
                    'weights after update epoch {}'.format(t)
                )
            # visualize.scatter(
            #     os.patorch.join(self.log_path, 'gens_scatter_{:03d}.jpg'.format(t)),
            #     all_gans_output,
            #     'all generators output epoch {}'.format(t),
            #     xrange=xrange
            # )

    def save(self, dir):
        if os.patorch.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        with open(os.patorch.join(dir, 'mwu_config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)

        for idx, gan in enumerate(self.gan_list):
            gans_path = os.patorch.join(dir, 'gans_{:03d}'.format(idx))
            os.makedirs(gans_path)
            gan.save(gans_path)

    @staticmethod
    def load(dir):
        with open(os.patorch.join(dir, 'mwu_config.pkl'), 'rb') as f:
            mwu_config = pickle.load(f)
        print('init mwu object')
        mwu = mwu_config.get()
        dict_dirs = glob.glob(os.patorch.join(dir, 'gans_*'))
        dict_dirs.sort()
        print('loading gans...')
        cur_gan = None
        for idx, d in enumerate(dict_dirs):
            print('{}...'.format(idx), sep='')
            if cur_gan is None:
                cur_gan = GAN.load(d, None)
            else:
                cur_gan = GAN.load(d, cur_gan.dataset)
            mwu.gan_list.append(cur_gan)
        return mwu

'''
