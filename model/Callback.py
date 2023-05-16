import numpy as np


class EarlyStoppingAtMinLoss:
    def __init__(self, patience=0):
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.wait = None
        self.stopped_epoch = None
        self.best = None
        self.best_epoch = None

    def on_train_begin(self):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
        self.best_epoch = 0

    def on_epoch_end(self, epoch, weights, loss=None):
        current = loss
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = weights
            self.best_epoch = epoch
            print(f' Epoch saved={self.best_epoch + 1}')
            return None
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                print("Restoring model weights from the end of the best epoch.")
                print(f'Best epoch={self.stopped_epoch + 1 - self.patience}')
                return self.best_weights
            print(' ')
            return None

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print("Epoch %4d: early stopping" % (self.stopped_epoch + 1))
        return self.best_weights


class LearningRateScheduler:
    def __init__(self, schedule=None, verbose=0):
        super().__init__()
        if schedule is None:
            self.schedule = self.lr_schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, model, backend):
        if not hasattr(model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(backend.get_value(model.optimizer.lr))
        lr = self.schedule(epoch, lr)
        backend.set_value(model.optimizer.learning_rate, lr)
        if self.verbose > 0:
            print(
                f"\nEpoch {epoch + 1}: LearningRateScheduler setting learning "
                f"rate to {lr}."
            )

    def on_epoch_end(self, model, backend, logs=None):
        logs = logs or {}
        logs["lr"] = backend.get_value(model.optimizer.lr)
        return logs

    def lr_schedule(self, epoch, lr):
        if epoch == 70:
            lr /= 2
        elif epoch == 50:
            lr *= .1
        elif epoch == 30:
            lr *= .1
        return lr
