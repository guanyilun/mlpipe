import torch
from torch.utils import data
import h5py
import numpy as np


class Dataset(data.Dataset):
    'Wrapper for TOD data PyTorch'
    def __init__(self, src, label):
        'Initialization'
        self._label = label
        self._hf = h5py.File(src, 'r', swmr=True)
        self._group = self._hf[label]
        self._keys = self._group.keys()

        self.param_keys = ['corrLive', 'rmsLive', 'kurtLive', 'DELive',
                           'MFELive', 'skewLive', 'normLive', 'darkRatioLive',
                           'jumpLive', 'gainLive']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self._keys)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        det_uid = self._keys[index]

        # Load data and get label
        dataset = self._group[det_uid]
        X = dataset[:]
        # sort in specific order
        y = dataset.attrs['label']

        # load pickle parameters
        params = np.zeros(len(self.param_keys))
        for i, key in enumerate(self.param_keys):
            params[i] = dataset.attrs[key]
        
        return X, params, y

    def get_sampler(self):
        n_keys = len(self._keys)
        
        labels = [self._group[self._keys[i]].attrs['label'] for i in range(n_keys)]
        n_good = np.sum(labels)
        n_bad = n_keys - n_good

        w_good = n_bad * 1.0 / n_keys
        w_bad = n_good * 1.0 / n_keys

        weights = map(lambda l: w_good if l == 0 else w_bad, labels)
        sampler = data.sampler.WeightedRandomSampler(weights, n_keys)
        return sampler
        



def truncate_collate(batch):
    """
    args:
        batch - list of (tensor, params, label)

    return:
        X - a tensor of all examples in 'batch' after padding
        params - a tensor of all pickle parameters
        y - a vector of labels
    """
    # find shortest sequence
    min_len = min([len(b[0]) for b in batch])

    # truncate according to min_len
    # stack all
    X = np.vstack(map(lambda x: x[0][:min_len], batch))
    params = np.vstack(map(lambda x: x[1], batch))
    y = np.array(map(lambda x: x[2], batch))
    return X, params, y
