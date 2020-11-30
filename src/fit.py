import torch


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
