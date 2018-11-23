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

  def __len__(self):
        'Denotes the total number of samples'
        return len(self._keys)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        det_uid = self._keys[index]

        # Load data and get label
        dataset = self._group[det_uid]
        X = dataset[:1000]
        y = dataset.attrs['label']
        return X, y
