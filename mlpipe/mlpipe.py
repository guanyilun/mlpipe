# -*- coding: utf-8 -*-

"""Main module."""

import torch
from torch.utils import data
import h5py
import numpy as np

class Dataset(data.Dataset):
    'Wrapper for TOD data PyTorch'
    def __init__(self, src, label):
        'Initialization'
        self._label = label
        self._hf = h5py.File(src, 'r')
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

