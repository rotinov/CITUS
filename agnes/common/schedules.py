from torch.optim.lr_scheduler import _LRScheduler


class LinearAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, eta_min=0.0, to_epoch=1000):
        self.eta_min = eta_min
        self.to_epoch = to_epoch
        last_epoch = -1
        super(LinearAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + max(0, (base_lr - self.eta_min) * (1 - self._step_count / self.to_epoch))
                for base_lr in self.base_lrs]

    def get_count(self):
        return self._step_count


class LinearSchedule:
    _step_count = 0

    def __init__(self, eta_min=0.0, to_epoch=1000):
        self.eta_min = eta_min
        self.to_epoch = to_epoch

    def step(self):
        self._step_count += 1

    def get_k(self):
        return self.eta_min + max(0.,
                                  (1. - self.eta_min) * (1. - self._step_count / self.to_epoch)
                                  )
