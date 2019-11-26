from gan import *
from dataset import *
import numpy as np
from matplotlib import pyplot as plt
import vis

def main():
    conf = {
        'gen_conf':{
            'name':'gan_g',
            'input_size': 10,
            'hidden_size': 32,
            'output_size': 2,
            'z_dim': 10
        },
        
        'dis_conf':{
            'name':'gan_d',
            'input_size': 2,
            'hidden_size': 64,
            'output_size': 1
        },
        'sampler':'gaussian'
    }

    import time
    start_t = time.time()
    dataset = Sine()

    print(len(dataset))

    import importlib
    import utils
    utils = importlib.reload(utils)

    dev = torch.device('cpu')
    mwugan = MWUGAN(conf, dataset=dataset, device=dev, gan_batchsize=128, update_method='mwu', delta=0.25)

    mwugan.g_step = 1
    mwugan.d_step = 1

    for i in range(20):
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

    results = []
    all_results = vis.show_synthetic_gan_all(mwugan.gan_list[:20], dataset)
    all_results = np.vstack(all_results)
    plt.clf()
    plt.scatter(all_results[:, 0], all_results[:, 1], alpha=0.005)
    plt.savefig("samples.png")
    plt.clf()
    plt.hist2d(all_results[:, 0], all_results[:, 1], bins=(100, 100), cmap=plt.cm.jet)
    plt.savefig("heatmap.png")

    return None, mwugan

def test_outlier_coverage(models):
    num = 500000
    num_each = num // len(models)
    all_results = []
    all_num = 0
    outlier_num = 0
    for i, model in enumerate(models):
        latent_samples = model.sampler.sampling(num_each)
        d_gen_input = latent_samples.to(dtype=torch.float32, device=model.device)
        results = model.gen(d_gen_input).to("cpu").detach().numpy()
        all_results.append(results)

        outlier_num += (((results - np.array([0, 8000])) ** 2).sum(axis=1) < 0.5 * 3).sum()
        all_num += num_each

    return outlier_num, all_num

from IPython import embed
from tqdm import tqdm
if __name__ == "__main__":
    mult_results = []
    models = []
    for _ in tqdm(range(1)):
        wanted, model = main()
        mult_results.append(wanted)
        models.append(model)
