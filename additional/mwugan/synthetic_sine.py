from gan import *
from dataset import *
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed

def test_outlier_coverage(models, i):
    num = 500000
    num_each = num // len(models)
    all_results = []
    all_num = 0
    outlier_num = 0
    for i, model in enumerate(models):
        latent_samples = model.sampler.sampling(num_each)
        d_gen_input = latent_samples.to(dtype=torch.float32, device=model.device)
        results = list(model.gen(d_gen_input).to("cpu").detach().numpy())
        all_results += results

        outlier_num += (((results - np.array([0, 8000])) ** 2).sum(axis=1) < 0.5 * 3).sum()
        all_num += num_each

    plt.clf()
    plt.scatter(np.vstack(all_results)[:, 0], np.vstack(all_results)[:, 1], alpha=0.05)
    plt.savefig("sine_sample_{}.png".format(i))
    np.savez("sine_sample_{}.npz".format(i), data=np.vstack(all_results))

    return outlier_num, all_num, np.vstack(all_results)

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
    plt.clf()
    plt.scatter(dataset.data[:, 0], dataset.data[:, 1], alpha=0.05)
    plt.savefig("sine_dataset.png")

    import importlib
    # import vis
    # vis = importlib.reload(vis)
    # _ = vis.show2ddataset(dataset)
    import utils
    utils = importlib.reload(utils)

    dev = torch.device('cpu')
    mwugan = MWUGAN(conf, dataset=dataset, device=dev, gan_batchsize=128, update_method='mwu', delta=0.25)

    mwugan.g_step = 1
    mwugan.d_step = 1

    for i in range(10):
        mwugan.iteration(dis_iters=20000, gan_iters=0)
        weights = mwugan.weights_list[-1]
        weights = np.reshape(weights, (-1, 50))
        weights = weights.mean(axis=1)
        print(weights)
        np.savez("weight_iter{}.npz".format(i), weight=weights)
        test_outlier_coverage(mwugan.gan_list, i)

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

    #results = []
    #for i in range(1, 25):
    #    all_results = vis.show_synthetic_gan_all(mwugan.gan_list[:i], dataset)
    #all_results = np.vstack(all_results)
    #results.append(test_unwanted(dataset.modes, dataset.sig, all_results))

    #return results, mwugan
    return None, mwugan


from IPython import embed
from tqdm import tqdm
if __name__ == "__main__":
    mult_results = []
    models = []
    for _ in tqdm(range(1)):
        wanted, model = main()
        mult_results.append(wanted)
        models.append(model)
        tmp1, tmp2, tmp3 = test_outlier_coverage(model.gan_list, 20)
    embed()
