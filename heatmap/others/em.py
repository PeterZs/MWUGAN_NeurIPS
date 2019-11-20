# -*- coding: utf-8 -*-
# @Author: yuchen
# @Date:   2019-05-07 10:45:08
# @Last Modified by:   yuchen
# @Last Modified time: 2019-07-27 16:35:47

from dataset import *
from sklearn import mixture
import numpy as np 
import matplotlib.pyplot as plt 
from IPython import embed
import time
time_base = 0
def tik():
    global time_base
    time_base = time.time()
    print("Start ticking")

def tok(name="(undefined)"):
    print("Time used in module {}: {}".format(name, time.time() - time_base))


dataset = Sine()
g = mixture.GaussianMixture(n_components=10)
tik()
g.fit(dataset.data)
tok("EM")

print(g.means_)
print(g.weights_)

plt.xlim(-12, 12)
plt.ylim(-12, 12)
idx = np.random.permutation(len(dataset.data))[:50000]
samples = dataset.data[idx].numpy()
plt.hist2d(list(samples[:, 0]) + [-12, 12] , list(samples[:, 1]) + [-12, 12], bins=(100, 100), cmap=plt.cm.jet)
plt.title("Dataset")
plt.colorbar()
plt.savefig("heatmap_dataset.pdf")
plt.clf()

print("Plotting dataset")
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.01, marker='.')
plt.xticks([-10, 10])
plt.yticks([-10, 10])
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.savefig("dataset.png", dpi=300)
plt.clf()


print("Generating samples")
samples = g.sample(125000)[0]

print("Plotting samples")
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.01)
plt.savefig("samples.png")
plt.clf()

samples[samples > 15] = 15
samples[samples < -15] = -15
samples = np.vstack((samples, np.array([[-15, -15], [15, 15]])))

plt.hist2d(samples[:, 0], samples[:, 1], bins=(100, 100), cmap=plt.cm.jet)
# plt.title("EM")
# plt.colorbar()
plt.axis('off')
plt.savefig("heatmap_em.png", dpi=300)
plt.clf()
# embed()



