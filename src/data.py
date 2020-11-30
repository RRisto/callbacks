import torch
from torch import tensor
from PIL import Image
import numpy as np


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype=np.int32)
    return data


def get_num_tensors(path):
    nums = sorted(list((path).glob('*.png')))
    num_tensors = [tensor(load_image(o)) for o in nums]
    stacked_num = torch.stack(num_tensors).float() / 255
    return stacked_num


def get_nums_tensors(root_path, nums=[3, 7]):
    stacked_train_tensors = []
    stacked_valid_tensors = []
    for num in nums:
        stacked_train_tensors.append(get_num_tensors(root_path / 'train' / str(num)))
        stacked_valid_tensors.append(get_num_tensors(root_path / 'valid' / str(num)))
    return stacked_train_tensors, stacked_valid_tensors


def tensors2dset(stacked_tensors):
    x = torch.cat(stacked_tensors).view(-1, 28 * 28)
    # assumes that we have only 2 levels of values in y (1 and 0)
    y = [i for i, sublist in enumerate(stacked_tensors) for item in sublist]
    y = tensor(y).unsqueeze(1)
    return x, y


def get_dsets(root_path, nums=[3, 7]):
    stacked_train_tensors, stacked_valid_tensors = get_nums_tensors(root_path, nums)
    train_x, train_y = tensors2dset(stacked_train_tensors)
    valid_x, valid_y = tensors2dset(stacked_valid_tensors)

    train_dset = list(zip(train_x, train_y))
    valid_dset = list(zip(valid_x, valid_y))
    return train_dset, valid_dset
