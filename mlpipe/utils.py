import torch
import numpy as np

def truncate_collate(batch):
    """
    args:
        batch - list of (tensor, label)

    return:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
    """
    # find shortest sequence
    min_len = min([len(b[0]) for b in batch])

    # truncate according to min_len
    # stack all
    X = np.vstack(map(lambda x: x[0][:min_len], batch))
    params = np.vstack(map(lambda x: x[1], batch))
    y = np.array(map(lambda x: x[2], batch))
    return X, params, y
