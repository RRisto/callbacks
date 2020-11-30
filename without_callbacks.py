import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

from src.data import get_dsets
from src.measure import validate_epoch, mnist_loss
from src.optim import BasicOptim


def one_batch_nocb(xb, yb, model, loss_func, opt):
    pred = model(xb)
    loss = loss_func(pred, yb)
    loss_train_value = loss.item()
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss_train_value


def fit_nocb(epochs, train_dl, valid_dl, model, loss_func, opt):
    for epoch in range(epochs):
        train_dl_tq = tqdm(train_dl, position=0, leave=True)
        for i, b in enumerate(train_dl):
            loss_train_value = one_batch_nocb(*b, model, loss_func, opt)
        acc_valid = validate_epoch(model, valid_dl)
        train_dl_tq.set_description(
            f"{epoch} loss train: {round(loss_train_value, 4)} valid acc: {round(acc_valid, 4)}")


if __name__ == '__main__':
    path = Path('data')
    train_dset, valid_dset = get_dsets(path)

    dl = DataLoader(train_dset, batch_size=256)
    valid_dl = DataLoader(valid_dset, batch_size=256)

    simple_net = nn.Sequential(
        nn.Linear(28 * 28, 30),
        nn.ReLU(),
        nn.Linear(30, 1))

    lr = 1e-3
    opt = BasicOptim(simple_net.parameters(), lr)

    fit_nocb(20, dl, valid_dl, simple_net, mnist_loss, opt)
