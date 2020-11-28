import numpy as np


class ReduceLROnPlateau:

    def __init__(self, patience=100, factor=0.5, min_lr=1e-4):
        # variables for learning rate scheduling
        self.performance_metric = 1e12
        self.wait = 0
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
    
    
    def update(self, cost, optimizer):
       
        if cost < self.performance_metric:
            # reset wait if performance metric improved
            self.performance_metric = cost
            self.wait = 0
        else:
            # increment wait if performance metric did not improve
            self.wait = self.wait + 1
            if (self.wait >= self.patience) and (optimizer._lr > self.min_lr):
                optimizer._lr = optimizer._lr * self.factor
                self.wait = 0
                print("  Learning rate decreased to {}".format(optimizer._lr))