
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-3):
        self.patience = patience
        self.delta = delta
        self.best = None
        self.count = 0
        self.stop = False
    def __call__(self, val):
        if self.best is None:
            self.best = val; return False
        if val < self.best - self.delta:
            self.best = val; self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True
        return self.stop
