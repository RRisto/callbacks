import matplotlib.pyplot as plt
from with_callbacks import *


## lr finder start here, pretty much is from https://fastai1.fast.ai/callbacks.lr_finder.html and https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
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

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, num_it: int = 100, stop_div: bool = True,
                 annealing_func=annealing_linear, beta=.98):
        self.stop_div = stop_div
        self.sched = Scheduler((start_lr, end_lr), num_it, annealing_func)
        self.lrs = []
        self.losses = []
        self.beta = beta
        self.avg_loss = 0
        self.iteration = 1

    def begin_fit(self):
        "Initialize optimizer and learner hyperparameters."
        self.opt = self.learn.opt
        self.opt.lr = self.sched.start
        self.stop, self.best_loss = False, 0.
        return True

    def after_loss(self, loss):
        "Determine if loss has runaway and we should stop."
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss.item()
        smooth_loss = self.avg_loss / (1 - self.beta ** self.iteration)

        if self.iteration == 0 or smooth_loss < self.best_loss: self.best_loss = smooth_loss
        self.lrs.append(self.opt.lr)
        self.losses.append(smooth_loss)
        self.opt.lr = self.sched.step()
        self.iteration += 1
        if self.sched.is_done or (self.stop_div and (smooth_loss > 4 * self.best_loss or torch.isnan(smooth_loss))):
            # We use the smoothed loss to decide on the stopping since it's less shaky.
            if not self.stop: self.stop = self.iteration
            return False
        return True

    def after_epoch(self):
        if self.stop: return False

    def after_fit(self):
        plt.plot(self.lrs[5:-5], self.losses[5:-5])
        plt.xscale('log')
        plt.show()


def lr_find(start_lr=1e-7, end_lr=10, num_it: int = 49, stop_div: bool = True, wd: float = None,
            annealing_func=annealing_exp):
    "Explore lr from `start_lr` to `end_lr` over `num_it` iterations in `learn`. If `stop_div`, stops when loss diverges."
    opt = BasicOptim(simple_net.parameters(), start_lr)
    cb = LRFinder(start_lr, end_lr, num_it, stop_div, annealing_func=annealing_func)
    learner = Learner(simple_net, mnist_loss, opt, dl, valid_dl, cb=CallbackHandler([cb]))
    epochs = int(np.ceil(num_it / len(learner.train_dl)))
    fit(epochs, learn=learner)


if __name__ == '__main__':
    path = Path('data')
    train_dset, valid_dset = get_dsets(path)

    dl = DataLoader(train_dset, batch_size=64)
    valid_dl = DataLoader(valid_dset, batch_size=256)

    simple_net = nn.Sequential(
        nn.Linear(28 * 28, 30),
        nn.ReLU(),
        nn.Linear(30, 1))

    lr_find(start_lr=1e-7, end_lr=5, num_it=len(dl), stop_div=True, wd=None, annealing_func=annealing_exp)
