class Learner:
    def __init__(self, model, loss_func, opt, train_dl, valid_dl, cb):
        self.model = model
        self.loss_func = loss_func
        self.opt = opt
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.cb = cb
        self.cb.set_learn(self)
