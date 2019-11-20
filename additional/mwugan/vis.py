from matplotlib import pyplot as plt
import torch
import torchvision.utils as vutils
import numpy as np
import utils

def show2ddataset(dataset, color='black', alpha=0.01):
    data = dataset.data
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    print(data.shape)
    plt.plot(data[:,0], data[:,1], 'o', color=color, alpha=alpha)
    # plt.show()

def show2dsinglegan(model, samples, alpha=0.1):
    latent_samples = model.sampler.sampling(samples)
    d_gen_input = latent_samples.to(dtype=torch.float32, device=model.device)
    d_fake_data = model.gen(d_gen_input)
    data = d_fake_data.detach().cpu().numpy()*100
    # print(d_fake_data.shape)
    plt.plot(data[:,0], data[:,1], 'o', alpha=alpha)

def show_synthetic_gan(model, dataset, num=5000):
    latent_samples = model.sampler.sampling(num)
    d_gen_input = latent_samples.to(dtype=torch.float32, device=model.device)
    results = model.gen(d_gen_input).to("cpu").detach().numpy()

    show2ddataset(dataset, color='black', alpha=0.01)

    plt.plot(results[:, 0], results[:, 1], 'o', color='blue', alpha=0.01)
    return results

def show_synthetic_gan_all(models, dataset, num=50000):
    num_each = num // len(models)
    show2ddataset(dataset, color='black', alpha=0.01)
    all_results = []
    for i, model in enumerate(models):
        latent_samples = model.sampler.sampling(num_each)
        d_gen_input = latent_samples.to(dtype=torch.float32, device=model.device)
        results = model.gen(d_gen_input).to("cpu").detach().numpy()
        plt.plot(results[:, 0], results[:, 1], 'o', color='blue', alpha=0.01)

        all_results.append(results)
    
    return all_results



def show_image_gan(model, n_row, n_col, normalize=True, ch=None):
    n_show = n_row * n_col
    latent_samples = model.sampler.sampling(n_show)
    d_gen_input = latent_samples.to(dtype=torch.float32, device=model.device)
    show_img = model.gen(d_gen_input)

    if ch is not None:
        show_img = show_img[:,ch-1:ch,:,:]

    fig=plt.figure(figsize=(30, 30), dpi=20, facecolor='w', edgecolor='k')
    showimg(vutils.make_grid(show_img.to("cpu").detach(), padding=1, normalize=normalize, nrow=n_row))
    return show_img

def showallimg(imgs, n_row, normalize=True):
    # fig=plt.figure(figsize=(30, 30), dpi=20, facecolor='w', edgecolor='k')
    showimg(vutils.make_grid(imgs.to("cpu").detach(), padding=1, normalize=normalize, nrow=n_row))

def showimg(img):
    if not isinstance(img, np.ndarray):
        npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

def show_image_dataset(dataset, n_row, n_col, normalize=True, augment=False, labels=False):
    # data = dataset.data
    n_show = n_row*n_col
    # print(data.shape)

    show_img, lb = dataset.sample_batch(n_show)
    if labels:
        print(lb)

    if augment:
        show_img = utils.random_noise(show_img, 0.1)

    # show_idx = np.random.choice(range(data.shape[0]), size=n_show, replace=False)
    # show_img = data[show_idx,:,:,:]

    fig=plt.figure(figsize=(30, 30), dpi=20, facecolor='w', edgecolor='k')
    if isinstance(show_img, np.ndarray):
        show_img = torch.FloatTensor(show_img)
    showimg(vutils.make_grid(show_img.to("cpu").detach(), padding=1, normalize=normalize, nrow=n_row))
    return show_img

def show_topk(element, show_size, normalize=True):
    split = len(element)//show_size
    idx = 0
    for i in range(split):
        fig=plt.figure(figsize=(30, 30), dpi=20, facecolor='w', edgecolor='k')
        show_img = torch.zeros(show_size, 1, 32, 32)
        for j in range(show_size):
            show_img[j,:,:,:] = element[idx][0]
            # (imgs[k,:,:,:], preds[k].item(), prob_max[k].item(), i, j, k)
            print('prediction', element[idx][1], 'probabitliy max', element[idx][2], 'gan id', element[idx][3], 'batch id', element[idx][4])
            idx += 1
        showimg(vutils.make_grid(show_img.to("cpu").detach(), padding=1, normalize=normalize, nrow=5))


def show_npy(npydata, n_row, n_col, normalize=True):
    n_show = n_row*n_col
    idx = np.random.choice(range(npydata.shape[0]), size=n_show)
    imgs = npydata[idx,:,:,:]
    show_img = torch.FloatTensor(imgs)
    showimg(vutils.make_grid(show_img.to("cpu").detach(), padding=1, normalize=normalize, nrow=n_row))


def testreload():
    print('reloaded')