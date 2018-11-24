# -*- coding: utf-8 -*-

"""Main module."""

import os
import torch

from data import Dataset, truncate_collate


class MLPipe(object):
    def __init__(self):
        self._epochs = 1
        self._models = dict()
        self.collate_fn = truncate_collate

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
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        loader_params = {
            'batch_size': 32,
            'shuffle': True,
            'num_workers': 4,
            'collate_fn': self.collate_fn
        }

        train_loader = DataLoder(self._train_set, **loader_params)
        test_loader = DataLoder(self._test_set, **loader_params))

        # setup all models
        for (k, model) in enumerate(self._models):
            model.setup(device)

        # train all models
        for epoch in range(self._epochs):
            for i, (batch, params, labels) in enumerate(train_loader):
                for (k, model) in enumerate(self._models):
                    metadata = {
                        'batch_id': i,
                        'params': params,
                        'device': device
                    }
                    model.train(batch, labels, metadata)

        # test all models
        criterion = nn.CrossEntropyLoss()
        for i, (batch, params, labels) in enumerate(test_loader):
            for (k, model) in enumerate(self._models):
                metadata = {
                    'batch_id': i,
                    'params': params,
                    'device': device
                }
                predictions = model.test(batch, labels, metadata)
                loss = criterion(predictions, labels)


        # clean up the memory
        for (k, model) in enumerate(self._models):
            model.cleanup()


class Model(object):

    name = ""

    def __init__(self):
        pass

    def setup(self, device):
        pass

    def train(self, batch, labels, metadata):
        raise RuntimeError("This method needs to be overridden!")

    def test(self, batch, labels, metadata):
        raise RuntimeError("This method needs to be overridden!")

    def cleanup(self):
        pass
