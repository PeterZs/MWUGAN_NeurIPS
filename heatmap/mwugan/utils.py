import torch
import numpy as np
#image aug
def random_noise(imgs, sig):
    noise = torch.randn(imgs.shape)*sig
    return imgs + noise

def sample_img_from_folder(base, n_samples, gan_id=None):
    if gan_id is None:
        gan_id = np.random.randint(0, high=49)
        print('gan id', gan_id)
    
    gan_folder = base+'gan{0:02d}'.format(gan_id)

    all_img = torch.zeros(30000, 1, 32, 32)
    for i in range(50):
        batch = torch.load(gan_folder+'/batch{0:03d}.pt'.format(i))
        all_img[600*i:600*(i+1), :, : ,:] = batch

    idx = np.random.choice(range(30000), size=n_samples, replace=True)
    return all_img[idx, :, :, :]


def eval_npy(filename, model, device, target, topk):
    top_data = []
    batch_size = 500
    idx = 0
    npdata = np.load(filename)
    print('np data size', npdata.shape)
    target_count = 0
    while 1:
        print((idx)*batch_size, '/', npdata.shape[0])
        if (idx+1)*batch_size>npdata.shape[0]:
            break
        imgs = npdata[idx*batch_size:(idx+1)*batch_size, :, :, :]
        idx += 1
        imgs = torch.from_numpy(imgs)
        imgs = imgs.to(dtype=torch.float32, device=device)
        log_probs = model(imgs)
        probs = torch.exp(log_probs)
        preds = log_probs.argmax(dim=1)
        prob_max = probs.max(dim=1)[0]
        for k in range(len(preds)):
            element = (imgs[k,:,:,:], preds[k].item(), prob_max[k].item(), idx, 0, k)
            top_data = insert_heap(top_data, element, topk, target)
            if(preds[k].item()==target):
                target_count+=1
    print(target_count)
    return top_data


def eval_all_from_folder(base, model, device, topk, startgan, endgan, n_batches, target):
    top_data = []
    for i in range(startgan, endgan):
        print('evaluating gan', i)
        path = base + 'gan{0:02d}/'.format(i)
        for j in range(n_batches):
            filename = path+'batch{0:03d}.pt'.format(j)
            data = torch.load(filename)
            imgs = data.to(dtype=torch.float32, device=device)
            log_probs = model(imgs)
            probs = torch.exp(log_probs)
            preds = log_probs.argmax(dim=1)
            prob_max = probs.max(dim=1)[0]
            for k in range(len(preds)):
                element = (imgs[k,:,:,:], preds[k].item(), prob_max[k].item(), i, j, k)
                top_data = insert_heap(top_data, element, topk, target)

    return top_data



def insert_heap(heap, val, k, target):
    if val[1] != target:
        return heap
    heap.append(val)
    heap.sort(key=lambda x:x[2])
    if(len(heap)>=k):
        heap = heap[1:]
    return heap