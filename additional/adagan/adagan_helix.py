# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""Training AdaGAN on various datasets.

Refer to the arXiv paper 'AdaGAN: Boosting Generative Models'
Coded by Ilya Tolstikhin, Carl-Johann Simon-Gabriel
"""

import os
import argparse
import logging
import tensorflow as tf
import numpy as np
from datahandler import DataHandler
from adagan import AdaGan
from metrics import Metrics
import utils

from IPython import embed

flags = tf.app.flags
flags.DEFINE_float("g_learning_rate", 0.01,
                   "Learning rate for Generator optimizers [16e-4]")
flags.DEFINE_float("d_learning_rate", 0.004,
                   "Learning rate for Discriminator optimizers [4e-4]")
flags.DEFINE_float("learning_rate", 0.008,
                   "Learning rate for other optimizers [8e-4]")
flags.DEFINE_float("adam_beta1", 0.5, "Beta1 parameter for Adam optimizer [0.5]")
flags.DEFINE_integer("zdim", 10, "Dimensionality of the latent space [100]")
flags.DEFINE_float("init_std", 0.8, "Initial variance for weights [0.02]")
flags.DEFINE_string("workdir", 'results_helix', "Working directory ['results']")
flags.DEFINE_bool("unrolled", False, "Use unrolled GAN training [True]")
flags.DEFINE_bool("is_bagging", False, "Do we want to use bagging instead of adagan? [False]")
FLAGS = flags.FLAGS

def main():
    opts = {}
    opts['random_seed'] = 821
    opts['dataset'] = 'helix' # gmm, circle_gmm,  mnist, mnist3, cifar ...
    opts['unrolled'] = FLAGS.unrolled # Use Unrolled GAN? (only for images)
    opts['unrolling_steps'] = 5 # Used only if unrolled = True
    opts['data_dir'] = 'mnist'
    opts['trained_model_path'] = 'models'
    opts['mnist_trained_model_file'] = 'mnist_trainSteps_19999_yhat' # 'mnist_trainSteps_20000'
    opts['gmm_max_val'] = 400.
    opts['toy_dataset_size'] = 64 * 1000
    opts['toy_dataset_dim'] = 2
    opts['mnist3_dataset_size'] = 2 * 64 # 64 * 2500
    opts['mnist3_to_channels'] = False # Hide 3 digits of MNIST to channels
    opts['input_normalize_sym'] = False # Normalize data to [-1, 1], applicable only for image datasets
    opts['adagan_steps_total'] = 25
    opts['samples_per_component'] = 5000 # 50000
    opts['work_dir'] = FLAGS.workdir
    opts['is_bagging'] = FLAGS.is_bagging
    opts['beta_heur'] = 'uniform' # uniform, constant
    opts['weights_heur'] = 'theory_star' # theory_star, theory_dagger, topk
    opts['beta_constant'] = 0.5
    opts['topk_constant'] = 0.5
    opts["init_std"] = FLAGS.init_std
    opts["init_bias"] = 0.0
    opts['latent_space_distr'] = 'normal' # uniform, normal
    opts['optimizer'] = 'sgd' # sgd, adam
    opts["batch_size"] = 64
    opts["d_steps"] = 1
    opts["g_steps"] = 1
    opts["verbose"] = True
    opts['tf_run_batch_size'] = 100
    opts['objective'] = 'JS'

    opts['gmm_modes_num'] = 10
    opts['latent_space_dim'] = FLAGS.zdim
    opts["gan_epoch_num"] = 200
    opts["mixture_c_epoch_num"] = 5
    opts['opt_learning_rate'] = FLAGS.learning_rate
    opts['opt_d_learning_rate'] = FLAGS.d_learning_rate
    opts['opt_g_learning_rate'] = FLAGS.g_learning_rate
    opts["opt_beta1"] = FLAGS.adam_beta1
    opts['batch_norm_eps'] = 1e-05
    opts['batch_norm_decay'] = 0.9
    opts['d_num_filters'] = 16
    opts['g_num_filters'] = 16
    opts['conv_filters_dim'] = 4
    opts["early_stop"] = -1 # set -1 to run normally
    opts["plot_every"] = 500 # set -1 to run normally
    opts["eval_points_num"] = 1000 # 25600
    opts['digit_classification_threshold'] = 0.999
    opts['inverse_metric'] = False # Use metric from the Unrolled GAN paper?
    opts['inverse_num'] = 1 # Number of real points to inverse.

    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    utils.create_dir(opts['work_dir'])
    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))

    data = DataHandler(opts)

    print("Data = {}".format(data.data.X))
    assert data.num_points >= opts['batch_size'], 'Training set too small'
    adagan = AdaGan(opts, data)
    metrics = Metrics()

    for step in range(opts["adagan_steps_total"]):
        logging.info('Running step {} of AdaGAN'.format(step + 1))

        adagan.make_step(opts, data)
        num_fake = opts['eval_points_num']
        logging.debug('Sampling fake points')
        fake_points = adagan.sample_mixture(num_fake)
        logging.debug('Sampling more fake points')
        more_fake_points = adagan.sample_mixture(50000)
        logging.debug('Plotting results')
        metrics.make_plots(opts, step, data.data[:500],
                fake_points[0:500], adagan._data_weights[:500])
        logging.debug('Evaluating results')

        (likelihood, C) = metrics.evaluate(
            opts, step, data.data[:500],
            fake_points, more_fake_points, prefix='')
    logging.debug("AdaGan finished working!")


from IPython import embed
from tqdm import tqdm
import numpy.random as nr

if __name__ == '__main__':
    def test_unwanted(modes, sigma, gendata):     
        wanted = 0                                                   
        unwanted = 0                         
        mode_covered = np.zeros((len(modes), ))       
        for point in gendata:           
            dists = ((modes - point) ** 2).sum(axis=1)
            if min(dists) < sigma * 3.0:       
                wanted += 1  
                mode_covered[np.argmin(dists)] += 1
            else:
                unwanted += 1
        return wanted, unwanted, (mode_covered > 0).sum()

    sigma = 1.0
    modes = np.array([[np.cos(i / 3) * i * i, np.sin(i / 3) * i * i] for i in range(20)])
    all_results = []
    for _ in tqdm(range(4)):
        print("Exp {} / 4".format(_ + 1))
        main()

        all_samples = []               
        results = []                    
        for i in range(25):                           
            print("Iter {} / 25".format(i + 1))
            a = np.load('results_helix/samples{:02d}.npy'.format(i))                   
            a = a[:, :, 0, 0]                    
            all_samples.append(a)                         
            tmp = np.vstack(all_samples)
            tmp = tmp[nr.choice(tmp.shape[0], 50000, replace=True)]
            results.append(test_unwanted(modes, sigma, tmp))
        all_results.append(results)

    embed()
                                


