"""
GQN training schedulers. Adapted from https://github.com/iShohei220/torch-gqn
"""

from torch.optim.lr_scheduler import _LRScheduler

class AnnealingStepLR(_LRScheduler):
    def __init__(self, optimizer, mu_i=5e-4, mu_f=5e-5, n=1.6e6):
        self.mu_i = mu_i
        self.mu_f = mu_f
        self.n = n
        super(AnnealingStepLR, self).__init__(optimizer)

    def get_lr(self):
        return [max(self.mu_f + (self.mu_i - self.mu_f) * (1.0 - self.last_epoch / self.n), self.mu_f) for base_lr in self.base_lrs]
