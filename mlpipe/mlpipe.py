# -*- coding: utf-8 -*-

"""Main module."""
from __future__ import print_function
import os
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from data import Dataset, truncate_collate

from report import Report


class MLPipe(object):
    def __init__(self):
        self._epochs = 1
        self._models = dict()
        self.collate_fn = truncate_collate
        self._param_keys = None

        # performance reporting
        self._report = Report()

        # internal counter for epoch and batch id
        self._epoch = 0  
        self._batch = 0

    def set_epochs(self, epochs):
        self._epochs = epochs

    def add_model(self, model):
        self._models[model.name] = model
        
    def set_dataset(self, src):
        if os.path.isfile(src):
            self._train_set = Dataset(src=src, label='train')
            self._validate_set = Dataset(src=src, label='validate')
            self._test_set = Dataset(src=src, label='test')

            # retrieve parameter keys
            self._param_keys = self._train_set.param_keys

        else:
            raise IOError("Dataset provided is not found!")

    def run(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # specify parameters used for train loader
        loader_params = {
            'batch_size': 128,
            'shuffle': True,
            'num_workers': 1,
            'collate_fn': self.collate_fn
        }
        train_loader = DataLoader(self._train_set, **loader_params)

        # setup all models
        for name in self._models.keys():
            model = self._models[name]
            model.setup(device)

        # train all models
        for epoch in range(self._epochs):
            self._epoch = epoch
            for i, (batch, params, label) in enumerate(train_loader):
                self._batch = i
                for name in self._models.keys():
                    model = self._models[name]
                    metadata = {
                        'batch_id': i,
                        'device': device
                    }
                    for idx, k in enumerate(self._param_keys):
                        metadata[k] = params[:, idx]

                    model.train(batch, label, metadata)

                if i % 100 == 0:
                    self.validate()
        
        # clean up the memory
        for name in self._models.keys():
            model = self._models[name]
            model.cleanup()

    def validate(self):        
        loader_params = {
            'batch_size': 128,
            'shuffle': False,
            'num_workers': 1,
            'collate_fn': self.collate_fn
        }
        validate_loader = DataLoader(self._validate_set, **loader_params)

        # initialize predictions dict
        predictions = {}
        for name in self._models.keys():
            predictions[name] = []

        # initialize labels list to store all labels
        labels = []
        for batch, params, label in self._validate_loader:
            labels.append(label)
            for name in self._models.keys():
                model = self._models[name]
                metadata = {
                    'batch_id': i,
                    'device': device
                }
                for idx, k in enumerate(self._param_keys):
                    metadata[k] = params[:, idx]

                prediction = model.validate(batch, label, metadata)

                predictions[name].append(prediction)

        # update performance dict and labels
        y_truth = np.vstack(labels)
        for name in self._models.keys():
            y_pred = np.vstack(predictions[name])
            self._report.add_record(name, self._epoch, self._batch, y_pred, y_truth)

        # print a intermediate result
        self._report.print_batch_report(self._epoch, self._batch)

    def test(self):
        # test all models
        loader_params = {
            'batch_size': 128,
            'shuffle': False,
            'num_workers': 1,
            'collate_fn': self.collate_fn
        }
        test_loader = DataLoader(self._test_set, **loader_params)

        for batch, params, label in enumerate(test_loader):
            for name in self._models.keys():
                model = self._models[name]
                metadata = {
                    'batch_id': i,
                    'device': device
                }
                for idx, k in enumerate(self._param_keys):
                    metadata[k] = params[:, idx]

                prediction = model.validate(batch, label, metadata)


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
