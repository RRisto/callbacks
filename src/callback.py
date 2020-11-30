import time
import numpy as np


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
