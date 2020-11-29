import torch
import torch.nn as nn
from torch import tensor
from PIL import Image
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
    def begin_fit(self):
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
    def begin_fit(self):
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
    def begin_fit(self):
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

    def set_learn(self, learn):
        self.learn = learn
        for cb in self.cbs:
            cb.learn = self.learn

    def begin_fit(self):
        self.in_train = True
        self.learn.stop = False
        res = True
        for cb in self.cbs: res = res and cb.begin_fit()
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


def fit(epochs, learn):
    if not learn.cb.begin_fit(): return
    for epoch in range(epochs):
        if not learn.cb.begin_epoch(epoch): continue
        all_batches(learn.train_dl, learn.cb, learn)  ###

        if learn.cb.begin_validate():
            with torch.no_grad(): all_batches(learn.valid_dl, learn.cb, learn)
        if learn.cb.do_stop() or not learn.cb.after_epoch(): break
    learn.cb.after_fit()


class Learner:
    def __init__(self, model, loss_func, opt, train_dl, valid_dl, cb):
        self.model = model
        self.loss_func = loss_func
        self.opt = opt
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.cb = cb
        self.cb.set_learn(self)


## lr finder

def annealing_linear(start, end, pct: float):
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start + pct * (end - start)


def annealing_exp(start, end, pct: float):
    "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start * (end / start) ** pct


def annealing_no(start, end, pct: float):
    "No annealing, always return `start`."
    return


def is_tuple(x) -> bool: return isinstance(x, tuple)


def is_listy(x) -> bool: return isinstance(x, (tuple, list))


class Scheduler():
    "Used to \"step\" from start,end (`vals`) over `n_iter` iterations on a schedule defined by `func`"

    def __init__(self, vals, n_iter: int, func=None):
        self.start, self.end = (vals[0], vals[1]) if is_tuple(vals) else (vals, 0)
        self.n_iter = max(1, n_iter)
        if func is None:
            self.func = annealing_linear if is_tuple(vals) else annealing_no
        else:
            self.func = func
        self.n = 0

    def restart(self):
        self.n = 0

    def step(self):
        "Return next value along annealed schedule."
        self.n += 1
        return self.func(self.start, self.end, self.n / self.n_iter)

    @property
    def is_done(self) -> bool:
        "Return `True` if schedule completed."
        return self.n >= self.n_iter


class LRFinder(Callback):
    "Causes `learn` to go on a mock training from `start_lr` to `end_lr` for `num_it` iterations."

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, num_it: int = 100,
                 stop_div: bool = True, annealing_func=annealing_linear, beta=.98):
        self.stop_div = stop_div
        self.sched = Scheduler((start_lr, end_lr), num_it, annealing_func)
        self.lrs = []
        self.losses = []
        self.beta = beta
        self.avg_loss = 0

    def begin_fit(self):
        "Initialize optimizer and learner hyperparameters."
        # setattr(pbar, 'clean_on_interrupt', True)
        # self.learn.save('tmp')
        self.iteration = 1
        self.opt = self.learn.opt
        self.opt.lr = self.sched.start
        self.stop, self.best_loss = False, 0.
        return {'skip_validate': True}

    def after_loss(self, loss, **kwargs):
        "Determine if loss has runaway and we should stop."
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss.item()
        smooth_loss = self.avg_loss / (1 - self.beta ** self.iteration)

        if self.iteration == 0 or smooth_loss < self.best_loss: self.best_loss = smooth_loss
        self.lrs.append(self.opt.lr)
        self.losses.append(smooth_loss)
        self.opt.lr = self.sched.step()
        if self.sched.is_done or (self.stop_div and (smooth_loss > 4 * self.best_loss or torch.isnan(smooth_loss))):
            # We use the smoothed loss to decide on the stopping since it's less shaky.
            if not self.stop: self.stop = self.iteration
            return False
        self.iteration += 1
        return True

    def after_epoch(self, **kwargs):
        if self.stop: return False

    def after_fit(self, **kwargs):
        "Cleanup learn model weights disturbed during LRFinder exploration."
        # self.learn.load('tmp', purge=False)
        if hasattr(self.learn.model, 'reset'): self.learn.model.reset()
        # for cb in self.cb:
        #     if hasattr(cb, 'reset'): cb.reset()
        print('LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.')
        # df_lr_loss = pd.DataFrame({'loss': self.losses[10:-5], 'lr': np.log(self.lrs[10:-5])})
        # df_lr_loss = df_lr_loss.set_index('lr')
        # df_lr_loss.loss.plot()
        # print(self.losses[10:-5])
        # print(self.lrs[10:-5])

        plt.plot(np.log10(self.lrs[5:-5]), self.losses[5:-5])
        plt.xticks(np.log10(self.lrs[5:-5]), self.lrs[5:-5])
       # ax = plt.gca()
        #ax.get_xaxis().set_major_formatter(plt.LogFormatter(10, labelOnlyBase=False))
        plt.show()


# total_batches = epoch * num_batch
# if total_batches - self.stop > 10:
#     print(
#         f"Best loss at batch #{self.stop}/{total_batches}, may consider .plot(skip_end={total_batches - self.stop + 3})")


def lr_find(start_lr=1e-7, end_lr=1, num_it: int = 49, stop_div: bool = True, wd: float = None,
            annealing_func=annealing_exp):
    "Explore lr from `start_lr` to `end_lr` over `num_it` iterations in `learn`. If `stop_div`, stops when loss diverges."
    # start_lr = learn.lr_range(start_lr)
    # start_lr = np.array(start_lr) if is_listy(start_lr) else start_lr
    # end_lr = learn.lr_range(end_lr)
    # end_lr = np.array(end_lr) if is_listy(end_lr) else end_lr
    opt = BasicOptim(simple_net.parameters(), start_lr)
    cb = LRFinder(start_lr, end_lr, num_it, stop_div, annealing_func=annealing_func)
    # todo make dynamic
    learner = Learner(simple_net, mnist_loss, opt, dl, valid_dl, cb=CallbackHandler([cb]))
    epochs = int(np.ceil(num_it / len(learner.train_dl)))
    fit(epochs, learn=learner)


# run

## Data
path = Path('data')
train_dset, valid_dset = get_dsets(path)

dl = DataLoader(train_dset, batch_size=64)
valid_dl = DataLoader(valid_dset, batch_size=256)

# model loss func, opt
simple_net = nn.Sequential(
    nn.Linear(28 * 28, 30),
    nn.ReLU(),
    nn.Linear(30, 1)
)

# lr = 1e-7
# opt = BasicOptim(simple_net.parameters(), lr)

# learner = Learner(simple_net, mnist_loss, opt, dl, valid_dl,
#                   cb=CallbackHandler([BatchCounter(), TimeCheck(), PrintLoss(), GetValAcc()]))

# fit(10, learn=learner, cb=CallbackHandler([BatchCounter(), TimeCheck(), PrintLoss(), PrintValidLoss()]))
# fit(10, learn=learner, cb=CallbackHandler([BatchCounter(), TimeCheck(), PrintLoss(), GetValAcc()]))
# fit(10, learn=learner)
# lr_find(start_lr=1e-7, end_lr=10, num_it=100, stop_div=True, wd=None)

lr_find(start_lr=1e-7, end_lr=1, num_it=49, stop_div=True, wd=None, annealing_func=annealing_exp)
