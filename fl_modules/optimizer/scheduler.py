import logging

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
import math
logger = logging.getLogger(__name__)

def change_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class GradualWarmupScheduler(LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, gamma: float, warmup_epochs: int, after_scheduler=None):
        self.gamma = gamma
        if self.gamma > 1.:
            raise ValueError('gamma should be less than 1.')
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.warmup_epochs:
            if self.after_scheduler:
                if not self.finished:
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            else:
                return [base_lr for base_lr in self.base_lrs]
        else: # in warmup
            return [base_lr * self.gamma + (1 - self.gamma) * base_lr * (self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        for key, value in self.after_scheduler.state_dict().items():
            if key != 'optimizer':
                state_dict['after_scheduler.' + key] = value
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict = {key: value for key, value in state_dict.items() if key != 'optimizer' and 'after_scheduler' not in key}
        super().load_state_dict(state_dict)
        after_scheduler_state_dict = {key.replace('after_scheduler.', ''): value for key, value in state_dict.items() if key.startswith('after_scheduler.')}
        if self.after_scheduler:
            self.after_scheduler.load_state_dict(after_scheduler_state_dict)

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        
        if self.last_epoch <= self.warmup_epochs:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epochs + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.warmup_epochs)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.warmup_epochs)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
            
class WarmupCosineAnnealingScheduler(LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, gamma: float, warmup_epochs: int, T_max: int, eta_min: float):
        self.gamma = gamma
        if self.gamma > 1.:
            raise ValueError('gamma should be less than 1.')
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_lrs = []
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.warmup_epochs:
            if self.last_epoch == self.warmup_epochs:
                self.last_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.last_lrs = [(1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / self.T_max)) / 
                             (1 + math.cos(math.pi * ((self.last_epoch - self.warmup_epochs) - 1) / self.T_max)) * 
                             (lr - self.eta_min) + self.eta_min for lr in self.last_lrs]
        else: # in warmup
            self.last_lrs = [base_lr * self.gamma + (1 - self.gamma) * base_lr * (self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]
        return self.last_lrs
