from torch.optim.lr_scheduler import _LRScheduler


class LinearAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a linear annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the scheduler init:

    .. math::
        \eta_t = \eta_{min} + (\eta_{max} - \eta_{min})(1 - \frac{T_{cur}}{T_{max}})

    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, eta_min=0.0, to_epoch=1000, last_epoch=-1):
        self.eta_min = eta_min
        self.to_epoch = to_epoch
        super(LinearAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + max(0, (base_lr - self.eta_min) * (1 - self._step_count / self.to_epoch))
                for base_lr in self.base_lrs]

    def get_count(self):
        return self._step_count
