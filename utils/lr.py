# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
@modified: zhulf0804 in 2019.11.22
"""
from bisect import bisect_right
import torch
from torch.optim import lr_scheduler


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def get_scheduler(cfg, optimizer):
    if cfg.OPTIM.LR == 'steplr':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.OPTIM.STEP_SIZE, gamma=cfg.OPTIM.GAMMA)
    elif cfg.OPTIM.LR =='warmup':
        scheduler = WarmupMultiStepLR(optimizer,
                                      milestones=cfg.OPTIM.MILSTONES,
                                      gamma=cfg.OPTIM.GAMMA,
                                      warmup_factor=cfg.OPTIM.WARMUP_FACTOR,
                                      warmup_iters=cfg.OPTIM.WARMUP_ITERS,
                                      warmup_method=cfg.OPTIM.WARMUP_METHOD,
                                      last_epoch=cfg.OPTIM.LAST_EPOCH)
    return scheduler