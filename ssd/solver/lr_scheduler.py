from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler
from math import cos, pi

class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, max_iter, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        self.max_iter = max_iter
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        lr_coeff =  (1 + cos(pi * self.last_epoch / self.max_iter)) / 2

        # return [
        #     base_lr
        #     * warmup_factor
        #     * (1 + cos(pi * (current_iter - self.warmup_iters) / (max_iter - self.warmup_iters))) / 2
        #     for base_lr in self.base_lrs
        # ]

        # print(self.warmup_factor, self.warmup_iters, self.last_epoch, self.max_iter)
        
        return [
            base_lr
            * warmup_factor
            * lr_coeff
            for base_lr in self.base_lrs
        ]

class WarmupCosLR(_LRScheduler):
    def __init__(self, optimizer, milestones, max_iter, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        self.max_iter = max_iter
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        lr_coeff =  (1 + cos(pi * self.last_epoch / self.max_iter)) / 2

        # return [
        #     base_lr
        #     * warmup_factor
        #     * (1 + cos(pi * (current_iter - self.warmup_iters) / (max_iter - self.warmup_iters))) / 2
        #     for base_lr in self.base_lrs
        # ]

        print(warmup_factor, warmup_iters, self.last_epoch, self.max_iter)
        
        return [
            base_lr
            * warmup_factor
            * lr_coeff
            for base_lr in self.base_lrs
        ]
