import numpy as np
from torch.utils.data import Dataset
from skimage import transform
import torch
import torchvision


class StackedMNIST(Dataset):
    def __init__(self, size, n_data, loadpath=None):
        self.size = size
        self.isimg = True
        self.n_data = n_data
        self.sample_weights = np.ones(self.n_data)
        if loadpath is not None:
            self.load(loadpath)
            self.n_data = self.data.shape[0]

            return

        trans = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(self.size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                       ])
        self.mnist_dataset = torchvision.datasets.MNIST(root='data/mnist2/', train=True, transform=trans, download=True)
        n_mnist = len(self.mnist_dataset)

        self.data = torch.zeros(n_data, 3, 32, 32).to(dtype=torch.float32)
        self.labels = torch.zeros(n_data).to(dtype=torch.long)
        self.n_data = n_data
        self.sample_weights = np.ones(self.n_data)

        idx0 = np.random.choice(range(n_mnist), size=self.n_data, replace=True)
        idx1 = np.random.choice(range(n_mnist), size=self.n_data, replace=True)
        idx2 = np.random.choice(range(n_mnist), size=self.n_data, replace=True)


        for i in range(self.n_data):
            if i%10000==0:
                print(i,'/',self.n_data)
            ch0, lb0 = self.mnist_dataset[idx0[i]]
            ch1, lb1 = self.mnist_dataset[idx1[i]]
            ch2, lb2 = self.mnist_dataset[idx2[i]]
            self.data[i, 0, :, :] = ch0
            self.data[i, 1, :, :] = ch1
            self.data[i, 2, :, :] = ch2
            self.labels[i] = lb0 + 10*lb1 + 100*lb2


        # ch0, lb0 = self.mnist_dataset[idx0]
        # ch1, lb1 = self.mnist_dataset[idx1]
        # ch2, lb2 = self.mnist_dataset[idx2]


    def __getitem__(self, index):
        return self.data[index,:,:,:], self.labels[index]

    def __len__(self):
        return self.n_data

    def sample_batch_uniform(self, batch_size):
        idx = np.random.choice(range(len(self.labels)), size=batch_size, replace=True)
        samples = self.data[idx, :, :, :]
        labels = self.labels[idx]
        return samples, labels

    def sample_batch(self, batch_size):
        p = self.sample_weights/np.sum(self.sample_weights)
        idx = np.random.choice(range(len(self.labels)), size=batch_size, replace=True, p=p)
        samples = self.data[idx, :, :, :]
        labels = self.labels[idx]
        return samples, labels

    def save(self, savepath):
        torch.save(self.data, savepath+"data.pth")
        torch.save(self.labels, savepath+"labels.pth")

    def load(self, loadpath):
        self.data = torch.load(loadpath+"data.pth")
        self.labels = torch.load(loadpath+"labels.pth")


class FasionMNIST_MNIST(Dataset):
    def __init__(self, size, FS_label_count, MNIST_label_count):
        self.size = size
        self.isimg = True

        trans = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(self.size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                       ])

        mnist_dataset = torchvision.datasets.MNIST(root='data/mnist2/', train=True, transform=trans, download=True)
        fashion_dataset = torchvision.datasets.FashionMNIST(root='data/fashionmnist/', train=True, transform=trans, download=True)

        assert(len(FS_label_count)==10)
        assert(len(MNIST_label_count)==10)
        
        # n_fs_data = len(fashion_dataset)
        # print(len(mnist_dataset))
        # print(len(fashion_dataset))

        self.fs_size = np.sum(FS_label_count)
        self.mnist_size = np.sum(MNIST_label_count)


        _fs_count = np.zeros(10)
        _mnist_count = np.zeros(10)
        self.images = torch.zeros(int(self.fs_size+self.mnist_size), 1, self.size, self.size)
        self.labels = torch.zeros(int(self.fs_size+self.mnist_size))

        c = 0
        for i in range(len(fashion_dataset)):
            img, label = fashion_dataset[i]
            lb = label.item()
            if _fs_count[lb]>=FS_label_count[lb]:
                continue
            _fs_count[lb]+=1
            self.images[c,:,:,:] = img
            self.labels[c] = label
            c += 1
        
        assert(np.sum(_fs_count)==self.fs_size)

        for i in range(len(mnist_dataset)):
            img, label = mnist_dataset[i]
            lb = label.item()
            if _mnist_count[lb]>=MNIST_label_count[lb]:
                continue
            _mnist_count[lb]+=1
            self.images[c,:,:,:] = img
            self.labels[c] = label
            c += 1

        assert(np.sum(_mnist_count)==self.mnist_size)

        self.sample_weights = np.ones(len(self.labels))

    def __getitem__(self, index):
        return self.images[index,:,:,:], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def sample_batch_uniform(self, batch_size):
        idx = np.random.choice(range(len(self.labels)), size=batch_size, replace=True)
        samples = self.images[idx, :, :, :]
        labels = self.labels[idx]
        return samples, labels

    def sample_batch(self, batch_size):
        p = self.sample_weights/np.sum(self.sample_weights)
        idx = np.random.choice(range(len(self.labels)), size=batch_size, replace=True, p=p)
        samples = self.images[idx, :, :, :]
        labels = self.labels[idx]
        return samples, labels


import cv2
from tqdm import tqdm
import os.path as op

class FashionMNIST2(Dataset):

    def __init__(self, loadpath=''):
        self.isimg = True

        datasetnpz = np.load(op.join(loadpath, 'MixedMNISTFull.npz'))
        tmp = datasetnpz['data']
        label = datasetnpz['label']

        x_train_28 = tmp.astype(np.float32).reshape([-1, 28, 28])
        n = x_train_28.shape[0]
        x_train_32 = np.zeros((n, 1, 32, 32))

        for i in tqdm(range(n)):
            x_train_32[i, 0] = cv2.resize(x_train_28[i], (32, 32)).astype('float32') / 255.

        x_train_32 = (x_train_32 - 0.5) / 0.5

        self.images = torch.FloatTensor(x_train_32)
        self.size = n

        assert (n == len(label))
        self.labels = torch.LongTensor(label)
        self.sample_weights = np.ones(len(self.labels))

    def __getitem__(self, index):
        return self.images[index,:,:,:], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def sample_batch_uniform(self, batch_size):
        idx = np.random.choice(range(len(self.labels)), size=batch_size, replace=True)
        samples = self.images[idx, :, :, :]
        labels = self.labels[idx]
        return samples, labels

    def sample_batch(self, batch_size):
        p = self.sample_weights/np.sum(self.sample_weights)
        idx = np.random.choice(range(len(self.labels)), size=batch_size, replace=True, p=p)
        samples = self.images[idx, :, :, :]
        labels = self.labels[idx]
        return samples, labels


class BiasedMNIST(Dataset):
    def __init__(self, size, label_count):
        self.size = size
        assert(len(label_count)==10)
        c_labels = np.zeros(10)
        self.isimg = True

        imagename = "data/MNIST/train-images-idx3-ubyte"
        labelname = "data/MNIST/train-labels-idx1-ubyte"
        n_img = 60000

        f = open(imagename, 'rb')
        l = open(labelname, 'rb')

        f.read(16)
        l.read(8)
        images = []
        labels = []

        for _ in range(n_img):
            lb = ord(l.read(1))
            img = []
            for j in range(28*28):
                img.append(ord(f.read(1)))
            if(c_labels[lb]>=label_count[lb]):
                continue
            img = np.reshape(np.array(img), (28, 28)).astype('uint8')
            img = transform.resize(img, (self.size, self.size))
            c_labels[lb] += 1
            labels.append(lb)
            images.append(np.expand_dims(img, axis=0))

        images = np.array(images)
        labels = np.array(labels)

        self.data = images
        self.label = labels
        # self.data = np.transpose(self.data, (0, 3, 1, 2))
        self.data = (self.data - 0.5) * 2
        # print(self.label)
        self.sample_weights = np.ones(len(labels))
        print('Biased MNIST generated:', c_labels)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

    
    def sample_batch_uniform(self, batch_size, to_tensor=True):
        idx = np.random.choice(range(len(self.label)), size=batch_size, replace=True)
        samples = self.data[idx, :, :, :]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels

    def sample_batch(self, batch_size, to_tensor=True):
        p = self.sample_weights/np.sum(self.sample_weights)
        idx = np.random.choice(range(len(self.label)), size=batch_size, replace=True, p=p)
        samples = self.data[idx, :, :, :]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels

    

class Helix(Dataset):
    def __init__(self, mode=20, sig=1.0, num_per_mode=50, hard=True):
        self.mode = mode
        self.num_per_mode = num_per_mode
        self.sig = sig
        self.length = num_per_mode * mode
        self.isimg = False
        if hard:
            self.modes = np.array([[np.cos(i / 3) * i * i, np.sin(i / 3) * i * i] for i in range(self.mode)])
        else:
            self.modes = np.array([[np.cos(i) * i, np.sin(i) * i] for i in range(self.mode)])
        data = None
        labels = []
        for loc in range(self.mode):
            if data is None:
                data = np.random.multivariate_normal((self.modes[loc, 0], self.modes[loc, 1]),
                                                     cov=[[self.sig, 0], [0, self.sig]], size=num_per_mode)
                labels.append(np.ones((num_per_mode, 1)) * loc)
            else:
                data = np.concatenate(
                    [
                        data,
                        np.random.multivariate_normal((self.modes[loc, 0], self.modes[loc, 1]),
                                                      cov=[[self.sig, 0], [0, self.sig]], size=num_per_mode)
                    ]
                )
                labels.append(np.ones((num_per_mode, 1)) * loc)
        self.data = torch.FloatTensor(data)
        self.label = torch.LongTensor(np.concatenate(labels, axis=0)[:, 0])
        self.sample_weights = np.ones(len(self.label))

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.length

    def sample_batch_uniform(self, batch_size, to_tensor=False):
        idx = np.random.choice(range(len(self.label)), size=batch_size, replace=True)
        samples = self.data[idx]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels

    def sample_batch(self, batch_size, to_tensor=False):
        p = self.sample_weights / np.sum(self.sample_weights)
        idx = np.random.choice(range(len(self.label)), size=batch_size, replace=True, p=p)
        samples = self.data[idx]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels


class GMM(Dataset):
    """Data set from AdaGAN GMM."""

    def __init__(self, mode=10, num_per_mode=64000, max_val=15):
        # np.random.seed(851)

        opts = {
            'gmm_modes_num': mode,  # 3
            'gmm_max_val': max_val,  # 15.
            'toy_dataset_dim': 2,
            'toy_dataset_size': num_per_mode  # 64 * 1000
        }

        modes_num = opts["gmm_modes_num"]
        max_val = opts['gmm_max_val']

        mixture_means = np.array(
            [[14.75207179, -9.5863695],
            [0.80064377, 3.41224097],
            [5.37076641, 11.76694952],
            [8.48660686, 11.73943841],
            [12.41315706, 2.82228677],
            [14.59626141, -2.52886563],
            [-7.8012091, 13.23184103],
            [-5.23725599, 6.27326752],
            [-6.87097889, 11.95825351],
            [10.79436725, -11.47316948]]
        )

        def variance_factor(num, dim):
            if num == 1: return 3 ** (2. / dim)
            if num == 2: return 3 ** (2. / dim)
            if num == 3: return 8 ** (2. / dim)
            if num == 4: return 20 ** (2. / dim)
            if num == 5: return 10 ** (2. / dim)
            return num ** 2.0 * 3

        mixture_variance = max_val / variance_factor(modes_num, opts['toy_dataset_dim'])

        num = opts['toy_dataset_size']
        X = np.zeros((num, opts['toy_dataset_dim']))
        labels = np.zeros((num))
        for idx in range(num):
            comp_id = np.random.randint(modes_num)
            mean = mixture_means[comp_id]
            cov = mixture_variance * np.identity(opts["toy_dataset_dim"])
            X[idx] = np.random.multivariate_normal(mean, cov, 1)
            labels[idx] = comp_id

        self.sig = mixture_variance
        self.modes = mixture_means
        self.data = torch.FloatTensor(X)
        self.label = torch.LongTensor(labels)
        self.sample_weights = np.ones(len(self.label))
        self.isimg = False

    def __getitem__(self, item):
        return self.data[item],self.label[item]

    def __len__(self):
        return len(self.data)

    def sample_batch_uniform(self, batch_size, to_tensor=False):
        idx = np.random.choice(range(len(self.label)), size=batch_size, replace=True)
        samples = self.data[idx]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels

    def sample_batch(self, batch_size, to_tensor=False):
        p = self.sample_weights / np.sum(self.sample_weights)
        idx = np.random.choice(range(len(self.label)), size=batch_size, replace=True, p=p)
        samples = self.data[idx]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels


class Outlier(Dataset):
    def __init__(self, sig=0.05, num_per_mode=50):
        self.num_per_mode = num_per_mode
        self.sig = sig
        self.isimg = False

        modex = np.arange(-10, 11)
        self.modes = []
        for y in modex:
            for x in modex:
                self.modes.append([y, x])

        self.modes.append([100, 100])
        self.modes = np.vstack(self.modes)

        data = None
        labels = []
        for loc in range(len(self.modes)):
            if data is None:
                data = np.random.multivariate_normal((self.modes[loc, 0], self.modes[loc, 1]),
                                                     cov=[[self.sig, 0], [0, self.sig]], size=num_per_mode)
                labels.append(np.ones((num_per_mode, 1)) * loc)
            else:
                data = np.concatenate(
                    [
                        data,
                        np.random.multivariate_normal((self.modes[loc, 0], self.modes[loc, 1]),
                                                      cov=[[self.sig, 0], [0, self.sig]], size=num_per_mode)
                    ]
                )
                labels.append(np.ones((num_per_mode, 1)) * loc)
        self.data = torch.FloatTensor(data)
        self.label = torch.LongTensor(np.concatenate(labels, axis=0)[:, 0])
        self.sample_weights = np.ones(len(self.label))

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.label)

    def sample_batch_uniform(self, batch_size, to_tensor=False):
        idx = np.random.choice(range(len(self.label)), size=batch_size, replace=True)
        samples = self.data[idx]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels

    def sample_batch(self, batch_size, to_tensor=False):
        p = self.sample_weights / np.sum(self.sample_weights)
        idx = np.random.choice(range(len(self.label)), size=batch_size, replace=True, p=p)
        samples = self.data[idx]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels


class Sine(Dataset):
    def __init__(self, dim=2, sig=0.5, num_per_mode=50):
        self.num_per_mode = num_per_mode
        self.sig = sig
        self.isimg = False
    
        uniform_num = 100000
        x = np.random.uniform(-10000, 10000, size=uniform_num)
        y = np.sin(x / 250 * np.pi) * x

        outlier_num = 250
        data2 = np.random.multivariate_normal((0, 10000), cov=np.diag([self.sig for _ in range(2)]).tolist(), size=outlier_num)

        data = np.vstack((x, y)).T
        data = np.vstack((data, data2))
        data /= 1000.
        # data[:, 1] -= 2000

        self.data = torch.FloatTensor(data)
        self.label = torch.LongTensor(np.zeros((uniform_num + outlier_num)))
        self.sample_weights = np.ones(len(self.label))

        self.n_data = len(self.data)
        self.maxval = np.abs(data).max()

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.label)

    def sample_batch_uniform(self, batch_size, to_tensor=False):
        idx = np.random.choice(range(len(self.label)), size=batch_size, replace=True)
        samples = self.data[idx]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels

    def sample_batch(self, batch_size, to_tensor=False):
        p = self.sample_weights / np.sum(self.sample_weights)
        idx = np.random.choice(range(len(self.label)), size=batch_size, replace=True, p=p)
        samples = self.data[idx]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels
