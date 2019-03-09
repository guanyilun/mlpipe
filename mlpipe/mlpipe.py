# -*- coding: utf-8 -*-
"""Main module."""
from __future__ import print_function
import time
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt

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
        self._output_dir = "outputs"

        # performance reporting
        self._report = Report(output_dir=self._output_dir)

        # internal counter for epoch and batch id
        self._epoch = 0
        self._batch = 0

        # sampling weight
        self._good_weight = 1
        self._bad_weight = 1
        self._has_setup = False

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

    def set_output_dir(self, output_dir):
        """Set output directory to store pipeline results"""
        self._output_dir = output_dir
        self._report.set_output_dir(output_dir)

    def load_dataset(self, src, load_data=True):
        if os.path.isfile(src):
            self._train_set = Dataset(src=src, label='train',
                                      load_data=load_data)
            self._validate_set = Dataset(src=src, label='validate',
                                         load_data=load_data)
            # In case that the test data is not available
            # do not stop but give a warining
            try:
                self._test_set = Dataset(src=src, label='test',
                                         load_data=load_data)
            except Exception:
                print("WARNING: test data is not available!")

            # retrieve parameter keys
            self._param_keys = self._train_set.param_keys

        else:
            raise IOError("Dataset provided is not found!")

    def setup(self):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print("GPU found, setting device to GPU...")
            device = torch.device("cuda:0")
        else:
            print("GPU not found or supported, setting device to CPU...")
            device = torch.device("cpu")

        # setup all models
        for name in self._models.keys():
            model = self._models[name]
            print("Setting up model: %s" % name)
            model.setup(device)

    def train(self, validate_plot=True):
        # automatically run setup if not ran already
        if not self._has_setup:
            self.setup()
            self._has_setup = True

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

        # train all models
        print("Training models...")
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
                    self.validate(plot=validate_plot)

    def validate(self, dataset=None, plot=True, test=False):
        # check the batch size specified, 0 means do not use
        # batch training
        if self._validate_batch_size == 0:
            batch_size = len(self._validate_set)
        else:
            batch_size = self._validate_batch_size

        loader_params = {
            'batch_size': batch_size,
            'shuffle': False,
            'collate_fn': self.collate_fn
        }

        if not dataset:
            validate_loader = DataLoader(self._validate_set, **loader_params)
        else:
            validate_loader = DataLoader(dataset, **loader_params)
        # initialize predictions to store the batch-wise predictions.
        # They will be merged together at the end. We also generate
        # a time_spent dictionary to store the time spend on each model
        # batch-wise. These will also be merged together at the end.
        predictions = {}
        probas = {}
        time_dict = {}
        for name in self._models.keys():
            predictions[name] = []
            probas[name] = []
            time_dict[name] = []

        # initialize labels list to store all labels
        labels = []
        if test:
            print("Testing models...")
        else:
            print("Validating models...")

        for batch, params, label in validate_loader:
            labels.append(label)
            for name in self._models.keys():
                model = self._models[name]
                metadata = {}
                for idx, k in enumerate(self._param_keys):
                    metadata[k] = params[:, idx][:, None]

                t_start = time.time()
                prediction, proba = model.validate(batch, label,
                                                   metadata)
                time_spent = time.time() - t_start

                # save the prediction and time spent in this batch
                predictions[name].append(prediction)
                probas[name].append(proba)
                time_dict[name].append(time_spent)

        # update performance dict and labels
        y_truth = np.hstack(labels)

        if test:
            epoch = -1
            batch = 0
            print("Saving testing data...")
        else:
            epoch = self._epoch
            batch = self._batch
            print("Saving validation data...")

        # Creating cross model roc and pr comparison plot
        if plot:
            roc_fig, roc_ax = plt.subplots(1,1)
            pr_fig, pr_ax = plt.subplots(1,1)
        else:
            roc_ax = pr_ax = None

        for name in self._models.keys():
            y_pred = np.hstack(predictions[name])
            y_pred_proba = np.vstack(probas[name])
            time_spent = sum(time_dict[name])
            self._report.add_record(name, epoch, batch,
                                    y_pred, y_pred_proba, y_truth,
                                    time_spent, plot=plot, roc_ax=roc_ax,
                                    pr_ax=pr_ax)

        if plot:
            # Save ROC curves
            roc_ax.set_title("ROC Curves")
            roc_ax.set_xlabel("False Positive Rate")
            roc_ax.set_ylabel("True Positive Rate")
            roc_ax.plot([0, 1], [0, 1], 'k--', lw=2)
            roc_ax.set_xlim([0.0, 1.0])
            roc_ax.set_ylim([0.0, 1.05])
            roc_ax.legend(loc='best', fontsize=12)
            filename = os.path.join(self._output_dir, "all_roc_curve.png")
            print("Saving plot: %s" % filename)
            roc_fig.savefig(filename)
            plt.close(roc_fig)

            # Save PR curves
            pr_ax.set_title("Precision-Recall Curves")
            pr_ax.set_xlabel("Recall")
            pr_ax.set_ylabel("Precision")
            pr_ax.set_xlim([0.0, 1.0])
            pr_ax.set_ylim([0.0, 1.05])
            pr_ax.legend(loc='best', fontsize=12)
            filename = os.path.join(self._output_dir, "all_pr_curve.png")
            print("Saving plot: %s" % filename)
            pr_fig.savefig(filename)
            plt.close(pr_fig)

        # print a intermediate result
        print('')
        if test:
            print('== TEST RESULTS: ==')
        else:
            print('== VALIDATION RESULTS: ==')

        self._report.print_batch_report(epoch, batch)

    def test(self):
        self.validate(dataset=self._test_set, test=True)

    def save(self):
        # create folder if not existing
        if not os.path.exists(self._output_dir):
            print('Path: {} does not exist, creating now ...'.format(self._output_dir))
            os.makedirs(self._output_dir)

        # save each model
        for name in self._models.keys():
            model = self._models[name]
            filename = os.path.join(self._output_dir, name+'.pickle')
            model.save(filename)

        # save report
        report_filename = os.path.join(self._output_dir, 'report.pickle')
        self._report.save()

    def load(self):
        """Load saved models data to continue training"""
        # try to load saved data
        if os.path.exists(self._output_dir):
            for name in self._models.keys():
                model = self._models[name]
                filename = os.path.join(self._output_dir, name+'.pickle')
                model.load(filename)

        # setup the models if not already setup
        if not self._has_setup:
            self.setup()
            self._has_setup = True

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

    def load(self, filename):
        pass

    def clean(self):
        pass
