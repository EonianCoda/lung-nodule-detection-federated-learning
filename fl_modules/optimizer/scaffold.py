import math

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required

class Scaffold(Optimizer):
    """Scaffold optimizer.

    Paper: https://arxiv.org/abs/1910.06378
    """

    def __init__(self,
                 params,
                 lr=required,
                 mu=0.0,
                 momentum=0,
                 dampening=0,
                 num_clients = 1,
                 weight_decay=0,
                 nesterov=False):
        """Initialize."""
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if lr is not required and lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if num_clients < 1:
            raise ValueError(f'Invalid num_clients value: {num_clients}')
        defaults = {
            'dampening': dampening,
            'lr': lr,
            'mu': mu,
            'num_clients': num_clients,
            'momentum': momentum,
            'nesterov': nesterov,
            'weight_decay': weight_decay,
        }

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError('Nesterov momentum requires a momentum and zero dampening')

        super(Scaffold, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        """Set optimizer state."""
        super(Scaffold, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            mu = group['mu']
            local_control_variate = group['local_control_variate']
            for p, local_control_variate_p in zip(group['params'], local_control_variate):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                # Lazy state initialization
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                param_state = self.state[p]
                if 'global_control_variate' not in param_state:
                    param_state['global_control_variate'] = torch.zeros_like(p, device=p.device)
                if 'local_control_variate' not in param_state:
                    param_state['local_control_variate'] = torch.zeros_like(p, device=p.device)
                        
                if local_control_variate_p is not None:
                    d_p.add_(param_state['global_control_variate'] - param_state['local_control_variate'], alpha=mu)
                p.add_(d_p, alpha=-group['lr'])

        return loss

    def update_global_weights(self):
        copy_params = [p.clone().detach().requires_grad_(False) for p in self.param_groups[0]['params']]
        for param_group in self.param_groups:
            param_group['w_old'] = copy_params
    
    @torch.no_grad()
    def update_control_variate(self):
        for group in self.param_groups:
            num_clients = group['num_clients']
            w_old = group['w_old']
            lr = group['lr']
            for p, w_old_p in zip(group['params'], w_old):
                param_state = self.state[p]
                global_control_variate = param_state['global_control_variate']
                local_control_variate = param_state['local_control_variate']
                # c_i = c_i - c + (1 / K * lr) * (w_old - w)
                global_control_variate.add_(local_control_variate - 2 * global_control_variate).add_(w_old_p - p, alpha=1 / num_clients * lr)
                local_control_variate.copy_(global_control_variate)

class ScaffoldAdam(Optimizer):
    """ScaffoldAdam optimizer."""

    def __init__(self, params, lr=1e-3, mu=0.0, num_clients = 1, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        """Initialize."""
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if mu < 0.0:
            raise ValueError(f'Invalid mu value: {mu}')
        if num_clients < 1:
            raise ValueError(f'Invalid num_clients value: {num_clients}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        defaults = {'lr': lr, 'betas': betas, 'eps': eps,
                    'weight_decay': weight_decay, 'amsgrad': amsgrad, 'mu': mu, 'num_clients': num_clients}
        super(ScaffoldAdam, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        """Set optimizer state."""
        super(ScaffoldAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            global_control_variates = []
            local_control_variates = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'Adam does not support sparse gradients, '
                            'please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = torch.tensor(0.0)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(
                                p, memory_format=torch.preserve_format)
                        state['global_control_variate'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['local_control_variate'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    global_control_variates.append(state['global_control_variate'])
                    local_control_variates.append(state['local_control_variate'])
                    
                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            self.adam(params_with_grad,
                      grads,
                      exp_avgs,
                      exp_avg_sqs,
                      max_exp_avg_sqs,
                      state_steps,
                      group['amsgrad'],
                      beta1,
                      beta2,
                      group['lr'],
                      group['weight_decay'],
                      group['eps'],
                      group['mu'],
                      local_control_variates,
                      global_control_variates,
                      )
        return loss

    def adam(self, params,
             grads,
             exp_avgs,
             exp_avg_sqs,
             max_exp_avg_sqs,
             state_steps,
             amsgrad,
             beta1: float,
             beta2: float,
             lr: float,
             weight_decay: float,
             eps: float,
             mu: float,
             local_control_variates,
             global_control_variates):
        """Updtae optimizer parameters."""
        for i, param in enumerate(params):
            grad = grads[i]
            grad.add_(global_control_variates[i] - local_control_variates[i], alpha=mu)
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)
            
    def update_global_weights(self):
        copy_params = [p.clone().detach().requires_grad_(False) for p in self.param_groups[0]['params']]
        for param_group in self.param_groups:
            param_group['w_old'] = copy_params
    
    @torch.no_grad()
    def update_control_variate(self):
        for group in self.param_groups:
            num_clients = group['num_clients']
            w_old = group['w_old']
            lr = group['lr']
            for p, w_old_p in zip(group['params'], w_old):
                param_state = self.state[p]
                global_control_variate = param_state['global_control_variate']
                local_control_variate = param_state['local_control_variate']
                # c_i = c_i - c + (1 / K * lr) * (w_old - w)
                global_control_variate.add_(local_control_variate - 2 * global_control_variate).add_(w_old_p - p, alpha=1 / num_clients * lr)
                local_control_variate.copy_(global_control_variate)