import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
import numpy.random as nr

class Outlier(Dataset):
    def __init__(self, dim=2, sig=0.05, num_per_mode=50):
        self.num_per_mode = num_per_mode
        self.sig = sig
        self.isimg = False

        """
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
        """
        uniform_num = 100000
        data1 = np.random.uniform(-10000, 10000, size=(uniform_num, dim))
        labels = [0 for _ in range(uniform_num)]

        outlier_num = 250
        data2 = np.random.multivariate_normal(tuple([4000 for _ in range(dim)]), cov=np.diag([self.sig for _ in range(dim)]).tolist(), size=outlier_num)
        labels += [1 for _ in range(outlier_num)]

        data = np.vstack((data1, data2))
        # data[:, 1] -= 2000
        data /= 2000
        self.data = torch.FloatTensor(data)
        self.label = torch.LongTensor(np.array(labels))
        self.sample_weights = np.ones(len(self.label))

        self.maxval = np.abs(data).max()

        self.n_data = len(self.data)
        self.order_idx = None
        self.reshuffle()


    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.n_data

    def sample_batch_uniform(self, batch_size, to_tensor=False):
        idx = np.random.choice(range(self.n_data), size=batch_size, replace=True)
        samples = self.data[idx]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels

    def sample_batch(self, batch_size, to_tensor=False):
        p = self.sample_weights / np.sum(self.sample_weights)
        idx = np.random.choice(range(self.n_data), size=batch_size, replace=True, p=p)
        samples = self.data[idx]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels

    def reshuffle(self):
        self.order_idx = nr.permutation(len(self.data))

    def next_batch(self, batch_size, start_idx):
        maxidx = min(start_idx + batch_size, self.n_data)
        samples = self.data[self.order_idx[start_idx:maxidx]]
        labels = self.label[self.order_idx[start_idx:maxidx]]
        return samples, labels


class Sine(Dataset):
    def __init__(self, dim=2, sig=0.05, num_per_mode=50):
        self.num_per_mode = num_per_mode
        self.sig = sig
    
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


    def reshuffle(self):
        self.order_idx = nr.permutation(len(self.data))

    def next_batch(self, batch_size, start_idx):
        maxidx = min(start_idx + batch_size, self.n_data)
        samples = self.data[self.order_idx[start_idx:maxidx]]
        labels = self.label[self.order_idx[start_idx:maxidx]]
        return samples, labels

