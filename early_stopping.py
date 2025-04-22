class EarlyStopper:

    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.min_val_loss = float('inf')
        self.best_model = None

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
            self.best_model = self.model.state_dict()
        else:
            self.counter += 1
        return self.counter >= self.patience