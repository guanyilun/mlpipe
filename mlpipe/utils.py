import torch

def truncate_collate(self, batch):
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
    X = torch.stack(map(lambda x: x[0][:min_len], batch), dim=0).type(torch.FloatTensor)
    params = torch.stack(map(lambda x: x[1], batch), dim=0).type(torch.FloatTensor)
    y = torch.FloatTensor(map(lambda x: x[2], batch))
    return X, params, y
