import torch
from torch.utils import data
import h5py
import numpy as np


class Dataset(data.Dataset):
    """Wrapper for TOD data PyTorch"""
    def __init__(self, src, label, load_data=True):
        """Initialization"""
        self._label = label
        self._hf = h5py.File(src, 'r', swmr=True)
        self._group = self._hf[label]
        self._keys = list(self._group)
        self._load_data = load_data
        self.param_keys = self._get_param_keys()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self._keys)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # select sample
        det_uid = self._keys[index]

        # load data and get label
        dataset = self._group[det_uid]

        # see if one needs data, if not X will simply be a placeholder
        if self._load_data:
            X = dataset[:]
        else:
            X = np.array([0])
        # sort in specific order
        y = dataset.attrs['label']

        # load pickle parameters
        params = np.zeros(len(self.param_keys))
        for i, key in enumerate(self.param_keys):
            params[i] = dataset.attrs[key]

        return X, params, y

    def _get_param_keys(self, scalp=0):
        """Retrieve a scalp data to get attrs"""
        det_uid = self._keys[scalp]

        # load data and get label
        dataset = self._group[det_uid]

        # get all parameters that are not "label" which is reserved
        # for truth
        return [par for par in dataset.attrs.keys() if par != "label"]

    def get_sampler(self, good=1, bad=1):
        n_keys = len(self._keys)

        good_bias = good*2.0/(good+bad)
        bad_bias = bad*2.0/(good+bad)
        labels = [self._group[self._keys[i]].attrs['label'] for i in range(n_keys)]
        n_good = np.sum(labels)
        n_bad = n_keys - n_good

        w_good = n_bad * 1.0 / n_keys * good_bias
        w_bad = n_good * 1.0 / n_keys * bad_bias

        weights = [(w_good if l == 1 else w_bad) for l in labels]
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
    min_len = min([b[0].shape[-1] for b in batch])
    # truncate according to min_len
    # stack all
    X = np.stack(list(map(lambda x: x[0][..., :min_len], batch)), axis=0)
    params = np.vstack(list(map(lambda x: x[1], batch)))
    y = [x[2] for x in batch]

    return X, params, y
