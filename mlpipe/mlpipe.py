# -*- coding: utf-8 -*-

"""Main module."""

import os
import torch
from torch.utils import data
import h5py
import numpy as np
import utils

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

class MLPipe(object):
    def __init__(self):
        self._epochs = 1
        self._models = dict()

    def set_epochs(self, epochs):
        self._epochs = epochs

    def add_model(model):
        self._models[model.name] = model
        
    def set_dataset(src):
        if os.path.isfile(src):
            self._train_set = Dataset(src=src, label='train')
            self._test_set = Dataset(src=src, label='test')
        else:
            raise IOError("Dataset provided is not found!")

    def run(self):
        loader_params = {
            'batch_size': 32,
            'shuffle': True,
            'num_workers': 4,
            'collate_fn': self._get_collate_fn()
        }
        train_loader = DataLoder(self._train_set, **loader_params)
        test_loader = DataLoder(self._test_set, **loader_params))

        # setup all models
        for (k, model) in enumerate(self._models):
            model.setup()

        # train all models
        for epoch in range(self._epochs):
            for i, (dets, params, labels) in enumerate(train_loader):
                for (k, model) in enumerate(self._models):
                    model.train(i, dets, params, labels)

        # test all models
        criterion = nn.CrossEntropyLoss()
        for i, (dets, params, labels) in enumerate(test_loader):
            for (k, model) in enumerate(self._models):
                predictions = model.test(i, dets, params, labels)
                loss = criterion(predictions, labels)

        # clean up the memory
        for (k, model) in enumerate(self._models):
            model.cleanup()

    def _get_collate_fn(self):
        return utils.truncate_collate

