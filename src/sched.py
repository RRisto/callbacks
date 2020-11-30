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
