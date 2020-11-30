import torch.nn as nn

from torch.utils.data import DataLoader
from pathlib import Path

from src.callback import CallbackHandler, BatchCounter, TimeCheck, PrintLoss, GetValAcc
from src.data import get_dsets
from src.fit import fit
from src.learner import Learner
from src.measure import mnist_loss
from src.optim import BasicOptim

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

    learner = Learner(simple_net, mnist_loss, opt, dl, valid_dl,
                      cb=CallbackHandler([BatchCounter(), TimeCheck(), PrintLoss(), GetValAcc()]))

    fit(10, learn=learner)
