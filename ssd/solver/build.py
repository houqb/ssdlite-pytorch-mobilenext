import torch

from .lr_scheduler import WarmupMultiStepLR, WarmupCosLR


def make_optimizer(cfg, model, lr=None):
    lr = cfg.SOLVER.BASE_LR if lr is None else lr
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)


def make_lr_scheduler(cfg, optimizer, max_iter, milestones=None):
    if cfg.SOLVER.LR_SCHEDULE == 'WarmupCosLR':
        return WarmupCosLR(optimizer=optimizer,
                             max_iter=max_iter,
                             warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                             warmup_iters=cfg.SOLVER.WARMUP_ITERS)
    else:
        return WarmupMultiStepLR(optimizer=optimizer,
                             milestones=cfg.SOLVER.LR_STEPS if milestones is None else milestones,
                             gamma=cfg.SOLVER.GAMMA,
                             warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                             warmup_iters=cfg.SOLVER.WARMUP_ITERS)
