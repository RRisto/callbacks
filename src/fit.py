import torch


def one_batch(xb, yb, learn):
    if not learn.cb.begin_batch(xb, yb): return
    loss = learn.cb.learn.loss_func(learn.cb.learn.model(xb), yb)
    if not learn.cb.after_loss(loss): return
    loss.backward()
    if learn.cb.after_backward(): learn.cb.learn.opt.step()
    if learn.cb.after_step(): learn.cb.learn.opt.zero_grad()


def all_batches(dl, learn):
    for xb, yb in dl:
        one_batch(xb, yb, learn)
        if learn.cb.do_stop(): return


def fit(epochs, learn):
    if not learn.cb.begin_fit(): return
    for epoch in range(epochs):
        if not learn.cb.begin_epoch(epoch): continue
        all_batches(learn.train_dl, learn)

        if learn.cb.begin_validate():
            with torch.no_grad(): all_batches(learn.valid_dl, learn)
        if learn.cb.do_stop() or not learn.cb.after_epoch(): break
    learn.cb.after_fit()
