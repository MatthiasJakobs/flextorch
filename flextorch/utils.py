import torch
import numpy as np

from sklearn.model_selection import train_test_split

def to_tensor(X, device='cpu'):
    if isinstance(X, torch.Tensor):
        return X.to(device)

    if isinstance(X, np.ndarray):
        return torch.from_numpy(X).to(device)

    raise NotImplementedError('Cannot convert type', type(X), 'to tensor')

def to_numpy(X):
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, torch.Tensor):
        return X.cpu().numpy()

    raise NotImplementedError('Cannot convert type', type(X), 'to np array')

def split_data(train_data, rng, percentage_train=0.8):
    if isinstance(train_data, torch.utils.data.DataLoader):
        raise RuntimeError('If you pass dataloaders as train data, you need to pass validation data aswell')
    if np.all([isinstance(obj, (torch.Tensor, np.ndarray)) for obj in train_data]):
        splits = train_test_split(*train_data, test_size=(1-percentage_train), random_state=rng) 
        return_train = []
        return_val = []

        for i in range(0, len(splits), 2):
            return_train.append(splits[0+i])
            return_val.append(splits[1+i])
        
        return tuple(return_train), tuple(return_val)

def deep_dict_update(old, new):
    for key in new.keys():
        if key in old.keys() and not isinstance(old[key], dict):
            old[key] = new[key]
        if key not in old.keys():
            old[key] = new[key]
        if key in old.keys() and isinstance(old[key], dict):
            _old, _new = deep_dict_update(old[key], new[key])
            old[key] = _old
            new[key] = _new

    return old, new