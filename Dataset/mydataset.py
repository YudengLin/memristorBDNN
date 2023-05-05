from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


class Dataset(Dataset):
    def __init__(self, root, train=True,to_tensor=True, preprocess_data = False):
        self.preprocess_data = preprocess_data
        if preprocess_data:
            self.Atmax, self.Atmin, self.Stmax, self.Stmin, self.Ymax, self.Ymin = np.loadtxt(root + 'bound.txt', ndmin=2).astype(np.float32)
            if train:
                X = np.loadtxt(root + 'X_train_pre.txt').astype(np.float32)
                Y = np.loadtxt(root + 'Y_train_pre.txt', ndmin=2).astype(np.float32)
            else:
                X = np.loadtxt(root + 'X_test_pre.txt').astype(np.float32)
                Y = np.loadtxt(root + 'Y_test_pre.txt', ndmin=2).astype(np.float32)
        else:
            if train:
                X = np.loadtxt(root + 'X_train.txt').astype(np.float32)
                Y = np.loadtxt(root + 'Y_train.txt', ndmin=2).astype(np.float32)
            else:
                X = np.loadtxt(root + 'X_test.txt').astype(np.float32)
                Y = np.loadtxt(root + 'Y_test.txt', ndmin=2).astype(np.float32)
        # Delta_t
        Y[:, :2] = Y[:, :2] - X[:, :2]

        if to_tensor:
            self.target = torch.from_numpy(Y).cuda()
            self.data = torch.from_numpy(X).cuda()
        else:
            self.target = Y
            self.data = X

    def plot_dist_dataset(self):
        pass

    def len(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.data.size(0)