from gan import *
from dataset import *
import numpy as np
from matplotlib import pyplot as plt
conf = {
    'gen_conf':{
        'name':'gan_g',
        'input_size': 10,
        'hidden_size': 16,
        'output_size': 2,
        'z_dim': 10
    },
    
    'dis_conf':{
        'name':'gan_d',
        'input_size': 2,
        'hidden_size': 32,
        'output_size': 1
    },
    'sampler':'gaussian'
}

import time
start_t = time.time()
dataset = GMM()

print(len(dataset))

import importlib
import vis
vis = importlib.reload(vis)
_ = vis.show2ddataset(dataset)
import utils
utils = importlib.reload(utils)

dev = torch.device('cpu')
mwugan = MWUGAN(conf, dataset=dataset, device=dev, gan_batchsize=128, update_method='mwu', delta=0.25)

mwugan.g_step = 1
mwugan.d_step = 1

for i in range(5):
    mwugan.iteration(dis_iters=20000, gan_iters=0)
    weights = mwugan.weights_list[-1]
    weights = np.reshape(weights, (-1, 50))
    weights = weights.mean(axis=1)
    print(weights)

    
from IPython import embed

def show_iter(it, num=5000):
    op = vis.show_synthetic_gan(mwugan.gan_list[it], dataset, num)
    plt.savefig("{}.png".format(it))
    plt.clf()

def test_unwanted(modes, sigma, gendata):
    wanted = 0
    unwanted = 0
    for point in gendata:
        dists = ((modes - point) ** 2).sum(axis=1)
        if min(dists) < sigma * 3.0:
            wanted += 1
        else:
            unwanted += 1
    return wanted, unwanted

embed()

all_results = vis.show_synthetic_gan_all(mwugan.gan_list, dataset)
all_results = np.vstack(all_results)
test_unwanted(dataset.modes, dataset.sig, all_results)
