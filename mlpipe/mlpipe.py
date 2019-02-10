# -*- coding: utf-8 -*-
"""Main module."""
from __future__ import print_function
import time
import os
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np

from .data import Dataset, truncate_collate
from .report import Report


class MLPipe(object):
    def __init__(self):
        self._epochs = 1
        self._train_batch_size = 128
        self._validate_batch_size = 128
        self._validate_interval = 100
        self._models = dict()
        self.collate_fn = truncate_collate
        self._param_keys = None

        # performance reporting
        self._report = Report()

        # internal counter for epoch and batch id
        self._epoch = 0  
        self._batch = 0

        # sampling weight
        self._good_weight = 1
        self._bad_weight = 1

    def add_model(self, model):
        self._models[model.name] = model

    def set_epochs(self, epochs):
        """Set the total number of epoches to run, and each epoch means running
        the entire training set once. For example, set_epochs(10) means all
        training data should run 10 times. This is only useful for models
        that support batch learning

        Params:
            epochs: integer, number of epochs to run
        """
        self._epochs = epochs

    def set_train_batch_size(self, batch_size):
        self._train_batch_size = int(batch_size)

    def set_validate_batch_size(self, batch_size):
        self._validate_batch_size = int(batch_size)

    def set_validate_interval(self, interval):
        """Set how often do we run validation during training. For example,
        a value of 100 (default value) means that after every 100 batches 
        we should run validation once. The result from the validation will
        be automatically saved in the report for future analysis. 

        Params:
            interval: integer, number of batches
        """

        self._validate_interval = int(interval)

    def set_train_bias(self, good, bad):
        """Set a bias between good and bad samples used in the training. 
        
        Examples:
            set_train_bias(good=1, bad=3) means that bad detectors are sampled
            three times good detectors. (good 25%; bad 75%)
        Params:
            good: weight for the good label
            bad: weight for the bad label
        """
        self._good_weight = good
        self._bad_weight = bad

    def load_dataset(self, src, load_data=True):
        if os.path.isfile(src):
            self._train_set = Dataset(src=src, label='train', 
                                      load_data=load_data)
            self._validate_set = Dataset(src=src, label='validate',
                                         load_data=load_data)
            self._test_set = Dataset(src=src, label='test',
                                     load_data=load_data)

            # retrieve parameter keys
            self._param_keys = self._train_set.param_keys

        else:
            raise IOError("Dataset provided is not found!")

    def train(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # check the batch size specified, 0 means do not use
        # batch training
        if self._train_batch_size == 0:
            batch_size = len(self._train_set)
        else:
            batch_size = self._train_batch_size
            
        # specify parameters used for train loader
        loader_params = {
            'batch_size': batch_size,
            'sampler': self._train_set.get_sampler(good=self._good_weight, 
                                                   bad=self._bad_weight),
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
                    metadata = {}
                    for idx, k in enumerate(self._param_keys):
                        metadata[k] = params[:, idx][:, None]
                    # time the training process
                    model.train(batch, label, metadata)

                if i % self._validate_interval == 0:
                    self.validate()

    def validate(self):
        loader_params = {
            'batch_size': self._validate_batch_size,
            'shuffle': False,
            'collate_fn': self.collate_fn
        }
        validate_loader = DataLoader(self._validate_set, **loader_params)

        # initialize predictions to store the batch-wise predictions.
        # They will be merged together at the end. We also generate
        # a time_spent dictionary to store the time spend on each model
        # batch-wise. These will also be merged together at the end. 
        predictions = {}
        time_dict = {}
        for name in self._models.keys():
            predictions[name] = []
            time_dict[name] = []

        # initialize labels list to store all labels
        labels = []
        for batch, params, label in validate_loader:
            labels.append(label)
            for name in self._models.keys():
                model = self._models[name]
                metadata = {}
                for idx, k in enumerate(self._param_keys):
                    metadata[k] = params[:, idx][:, None]

                t_start = time.time()
                prediction = model.validate(batch, label, metadata)
                time_spent = time.time() - t_start

                # save the prediction and time spent in this batch
                predictions[name].append(prediction)
                time_dict[name].append(time_spent)


        # update performance dict and labels
        y_truth = np.hstack(labels)
        for name in self._models.keys():
            y_pred = np.hstack(predictions[name])
            time_spent = sum(time_dict[name])
            self._report.add_record(name, self._epoch, self._batch, y_pred, y_truth, time_spent)

        # print a intermediate result
        print('== VALIDATION RESULTS: ==')        
        self._report.print_batch_report(self._epoch, self._batch)

    def test(self):
        loader_params = {
            'batch_size': self._validate_batch_size,
            'shuffle': False,
            'collate_fn': self.collate_fn
        }
        test_loader = DataLoader(self._test_set, **loader_params)

        # initialize predictions and time dict. similar to validation,
        # we will save the result batch-wise
        predictions = {}
        time_dict = {}
        for name in self._models.keys():
            predictions[name] = []
            time_dict[name] = []

        # initialize labels list to store all labels
        labels = []
        for batch, params, label in test_loader:
            labels.append(label)
            for name in self._models.keys():
                model = self._models[name]
                metadata = {}
                for idx, k in enumerate(self._param_keys):
                    metadata[k] = params[:, idx][:, None]

                t_start = time.time()
                prediction = model.validate(batch, label, metadata)
                time_spent = time.time() - t_start

                predictions[name].append(prediction)
                time_dict[name].append(time_spent)

        # update performance dict and labels
        y_truth = np.hstack(labels)
        for name in self._models.keys():
            y_pred = np.hstack(predictions[name])
            time_spent = sum(time_dict[name])
            self._report.add_record(name, -1, 0, y_pred, y_truth, time_spent)

        # print a intermediate result
        print('== TEST RESULTS: ==')
        self._report.print_batch_report(-1, 0)

    def save(self, path):
        # create folder if not existing
        if not os.path.exists(path):
            print('Path: {} does not exist, creating now ...'.format(path))
            os.makedirs(path)

        # save each model
        for name in self._models.keys():
            model = self._models[name]
            filename = os.path.join(path, name+'.pickle')
            model.save(filename)

        # save report
        report_filename = os.path.join(path, 'report.pickle')
        self._report.save(report_filename)

    def clean(self):
        for name in self._models.keys():
            model = self._models[name]
            model.clean()


class Model(object):

    name = ""

    def __init__(self):
        pass

    def setup(self, device):
        pass

    def train(self, batch, labels, metadata):
        raise RuntimeError("This method needs to be overridden!")

    def validate(self, batch, labels, metadata):
        raise RuntimeError("This method needs to be overridden!")

    def save(self, filename):
        pass

    def clean(self):
        pass
