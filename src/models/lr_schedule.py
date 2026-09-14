"""
Linear warmup then cosine decay, stepped once per iteration. Same schedule as
Dinomaly's WarmCosineScheduler (utils.py), rewritten without the _LRScheduler
plumbing so it's just an index into a precomputed array.
"""

import math


class WarmCosineSchedule:
    def __init__(self, optimizer, base_lr, final_lr, total_iters, warmup_iters=100):
        self.optimizer = optimizer
        self.final_lr = final_lr
        self.total_iters = total_iters

        warmup = [base_lr * i / max(warmup_iters, 1) for i in range(warmup_iters)]
        cosine = [
            final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * i / (total_iters - warmup_iters)))
            for i in range(total_iters - warmup_iters)
        ]
        self.schedule = warmup + cosine
        self._it = 0

    def step(self):
        lr = self.schedule[self._it] if self._it < len(self.schedule) else self.final_lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._it += 1
        return lr


class WarmupCosineRatioSchedule:
    """
    Dinomaly2's WarmupCosineScheduler (utils.py), stepped the same way as
    WarmCosineSchedule above. Unlike v1's schedule, each param group keeps its
    own base lr (Dinomaly2 gives the bottleneck's first layer a lower lr than
    the rest) and is scaled by the same warmup/cosine *ratio* rather than all
    groups sharing one absolute value. `final_ratio` is relative to each
    group's own base lr; Dinomaly2's own MVTec-AD/VisA commands use the
    default final_ratio=1.0, i.e. warmup then hold constant, no decay.
    """

    def __init__(self, optimizer, total_iters, warmup_iters=100, final_ratio=1.0):
        self.optimizer = optimizer
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.final_ratio = final_ratio
        self._it = 0

    def _ratio(self, it):
        if it < self.warmup_iters:
            return it / max(self.warmup_iters, 1)
        progress = (it - self.warmup_iters) / max(self.total_iters - self.warmup_iters, 1)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_ratio + (1.0 - self.final_ratio) * coeff

    def step(self):
        ratio = self._ratio(min(self._it, self.total_iters))
        lrs = [base * ratio for base in self.base_lrs]
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        self._it += 1
        return lrs
