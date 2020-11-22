import torch
import torch.nn as nn
from torch import tensor
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import time


## helpers
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
        stacked_train_tensors.append(get_num_tensors(path / 'train' / str(num)))
        stacked_valid_tensors.append(get_num_tensors(path / 'valid' / str(num)))
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


## Data
path = Path('data')
train_dset, valid_dset = get_dsets(path)

dl = DataLoader(train_dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)

# model loss func, opt
simple_net = nn.Sequential(
    nn.Linear(28 * 28, 30),
    nn.ReLU(),
    nn.Linear(30, 1)
)


def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()


class BasicOptim:
    def __init__(self, params, lr):
        self.params, self.lr = list(params), lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None


lr = 1e-3
opt = BasicOptim(simple_net.parameters(), lr)


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()


def validate_epoch(model, valid_dl):
    model.eval()
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    model.train()
    return round(torch.stack(accs).mean().item(), 4)


def one_batch(xb, yb, model, loss_func, opt):
    pred = model(xb)
    loss = loss_func(pred, yb)
    loss_train_value = loss.item()
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss_train_value


def fit(epochs, train_dl, valid_dl, model, loss_func, opt):
    for epoch in range(epochs):
        train_dl_tq = tqdm(train_dl, position=0, leave=True)
        for i, b in enumerate(train_dl):
            loss_train_value = one_batch(*b, model, loss_func, opt)
        acc_valid = validate_epoch(model, valid_dl)
        train_dl_tq.set_description(
            f"{epoch} loss train: {round(loss_train_value, 4)} valid acc: {round(acc_valid, 4)}")


fit(20, dl, valid_dl, simple_net, mnist_loss, opt)
