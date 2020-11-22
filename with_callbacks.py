import torch
import torch.nn as nn
from torch import tensor
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


# callbacks
class Callback():
    def begin_fit(self, learn):
        self.learn = learn
        return True

    def after_fit(self): return True

    def begin_epoch(self, epoch):
        self.epoch = epoch
        return True

    def begin_validate(self): return True

    def after_epoch(self): return True

    def begin_batch(self, xb, yb):
        self.xb, self.yb = xb, yb
        return True

    def after_loss(self, loss):
        self.loss = loss
        return True

    def after_backward(self): return True

    def after_step(self): return True


class BatchCounter(Callback):
    def begin_epoch(self, epoch):
        self.epoch = epoch
        self.batch_counter = 1
        return True

    def after_step(self):
        self.batch_counter += 1
        if self.batch_counter % 200 == 0: print(f'Batch {self.batch_counter} completed')
        return True


class TimeCheck(Callback):
    def begin_fit(self, learn):
        self.learn = learn
        self.epoch_counter = 1
        return True

    def begin_epoch(self, epoch):
        self.epoch = epoch
        print(f'Epoch {self.epoch_counter} started at {time.strftime("%H:%M:%S", time.gmtime())}')
        self.epoch_counter += 1
        return True


class PrintLoss(Callback):
    def after_epoch(self):
        print(f'Loss train: {round(self.loss.item(), 4)}')
        return True


class PrintValidLoss(Callback):
    def __init__(self):
        self.in_train = True

    def begin_validate(self):
        self.in_train = False
        self.val_losses = []
        return True

    def begin_batch(self, xb, yb):
        if not self.in_train:
            super(PrintValidLoss, self).begin_batch(xb, yb)
        return True

    def after_loss(self, loss):
        if not self.in_train:
            self.val_losses.append(loss.item())
        return False

    def after_epoch(self):
        if not self.in_train:
            print(f'Loss valid: {self.val_losses}')
            self.in_train = True
        return True


class GetValAcc(Callback):
    def begin_fit(self, learn):
        self.learn = learn
        self.in_train = True
        return True

    def batch_accuracy(self, xb, yb):
        xb = self.learn.model(xb)
        preds = xb.sigmoid()
        correct = (preds > 0.5) == yb
        return correct.float().mean().item()

    def begin_validate(self):
        self.in_train = False
        self.accs = []
        return True

    def begin_batch(self, xb, yb):
        if not self.in_train:
            acc = self.batch_accuracy(xb, yb)
            self.accs.append(acc)
        return True

    def after_epoch(self):
        if not self.in_train:
            print(f'Valid accuracy: {round(np.mean(self.accs), 4)}')
            self.in_train = True
        return True


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()


def validate_epoch(model, valid_dl):
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)


class CallbackHandler():
    def __init__(self, cbs=None):
        self.cbs = cbs if cbs else []

    def begin_fit(self, learn):
        self.learn, self.in_train = learn, True
        self.learn.stop = False
        res = True
        for cb in self.cbs: res = res and cb.begin_fit(learn)
        return res

    def after_fit(self):
        res = not self.in_train
        for cb in self.cbs: res = res and cb.after_fit()
        return res

    def begin_epoch(self, epoch):
        self.learn.model.train()
        self.in_train = True
        res = True
        for cb in self.cbs: res = res and cb.begin_epoch(epoch)
        return res

    def begin_validate(self):
        self.learn.model.eval()
        self.in_train = False
        res = True
        for cb in self.cbs: res = res and cb.begin_validate()
        return res

    def after_epoch(self):
        res = True
        for cb in self.cbs: res = res and cb.after_epoch()
        return res

    def begin_batch(self, xb, yb):
        res = True
        for cb in self.cbs: res = res and cb.begin_batch(xb, yb)
        return res

    def after_loss(self, loss):
        res = self.in_train
        for cb in self.cbs:
            res = res and cb.after_loss(loss)
        return res

    def after_backward(self):
        res = True
        for cb in self.cbs: res = res and cb.after_backward()
        return res

    def after_step(self):
        res = True
        for cb in self.cbs: res = res and cb.after_step()
        return res

    def do_stop(self):
        try:
            return self.learn.stop
        finally:
            self.learn.stop = False


def one_batch(xb, yb, cb, learn):
    if not cb.begin_batch(xb, yb): return
    loss = cb.learn.loss_func(cb.learn.model(xb), yb)
    if not cb.after_loss(loss): return
    loss.backward()
    if cb.after_backward(): cb.learn.opt.step()
    if cb.after_step(): cb.learn.opt.zero_grad()


def all_batches(dl, cb, learn):
    for xb, yb in dl:
        one_batch(xb, yb, cb, learn)
        if cb.do_stop(): return


def fit(epochs, learn, cb):
    if not cb.begin_fit(learn): return
    for epoch in range(epochs):
        if not cb.begin_epoch(epoch): continue
        all_batches(learn.train_dl, cb, learn)  ###

        if cb.begin_validate():
            with torch.no_grad(): all_batches(learn.valid_dl, cb, learn)
        if cb.do_stop() or not cb.after_epoch(): break
    cb.after_fit()


class Learner:
    def __init__(self, model, loss_func, opt, train_dl, valid_dl):
        self.model = model
        self.loss_func = loss_func
        self.opt = opt
        self.train_dl = train_dl
        self.valid_dl = valid_dl


train_dset, valid_dset = get_dsets(path)

dl = DataLoader(train_dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)

simple_net = nn.Sequential(
    nn.Linear(28 * 28, 30),
    nn.ReLU(),
    nn.Linear(30, 1)
)

lr = 1e-3
opt = BasicOptim(simple_net.parameters(), lr)

learner = Learner(simple_net, mnist_loss, opt, dl, valid_dl)

# fit(10, learn=learner, cb=CallbackHandler([BatchCounter(), TimeCheck(), PrintLoss(), PrintValidLoss()]))
fit(10, learn=learner, cb=CallbackHandler([BatchCounter(), TimeCheck(), PrintLoss(), GetValAcc()]))
# fit(10, learn=learner, cb=CallbackHandler([BatchCounter(), TimeCheck(), PrintLoss()]))
