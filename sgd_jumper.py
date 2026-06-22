import torch
from torch.optim import Optimizer
import math

def triangle_wave(t, T=1.0, max=1.0 , min=0.1):
    x = t % T
    if x < T/2:
        return min + (max - min) * (2 * x / T)
    else:
        return min + (max - min) * (2 - 2 * x / T)


class Jumper(Optimizer):
    def __init__(self, params,steps_per_epoch, lr=0.5, ocilation=1e-2, momentum=0.0, jump_mult=1.5,fit_type='linear',weight_decay=0.0):
        """
        SGD-Jumper: Trend extrapolation of weights.
        
        The optimizer fits a linear or any a.f(x) + b trend to weight updates within an epoch 
        and 'jumps' forward at the end of the epoch.
        """
        print(f"Initializing Jumper with steps_per_epoch={steps_per_epoch}, lr={lr}, ocilation={ocilation}, momentum={momentum}, jump_mult={jump_mult}, weight_decay={weight_decay}")
        if fit_type == 'linear':
            self.activation = lambda x: x
        elif fit_type == 'log':
            self.activation = x
        elif fit_type == 'sqrt':
            self.activation = lambda x: math.sqrt(x)
        else:
            raise ValueError(f"Invalid fit_type: {fit_type}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
    
        defaults = dict(lr=lr, ocilation=ocilation, momentum=momentum, jump_mult=jump_mult, steps_per_epoch=steps_per_epoch, weight_decay=weight_decay)
        super(Jumper, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                self._init_param_state(p, state)

    def _init_param_state(self, p, state):
        state['step_count'] = 0
        state['sum_x'] = 0.0
        state['sum_y'] = torch.zeros_like(p.data)
        state['sum_xy'] = torch.zeros_like(p.data)
        state['sum_x2'] = 0.0
        state['momentum_buffer'] = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            ocilation = group['ocilation']
            lr_min = lr * ocilation
            steps_per_epoch = group['steps_per_epoch']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                d_p = p.grad

                # 1. Momentum Update with Reset logic
                if momentum != 0:
                    buf = state.get('momentum_buffer')
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        state['momentum_buffer'] = buf
                    else:
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                
                # 2. Bias Correction & Weight Update
                # This helps stabilize the 'recovery' phase after a jump
                bias_correction = 1 - momentum ** (state['step_count'] + 1)
                lr_effective = triangle_wave(state['step_count'], steps_per_epoch, lr, lr_min)
                p.data.add_(d_p, alpha=-lr_effective * bias_correction)

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr_effective)

                # 3. Accumulate Linear Regression Stats
                x =  self.activation(float(state['step_count']))
                y = p.data.detach()

                state['sum_x'] += x
                state['sum_y'].add_(y)
                state['sum_xy'].add_(y.mul(x))
                state['sum_x2'] += x**2
                state['step_count'] += 1

        return loss

    @torch.no_grad()
    def jump(self):
        """Fits ax+b to weight history and projects to n = steps * jump_mult"""
        last_n_target = 0
        for group in self.param_groups:
            jump_mult = group['jump_mult']
            jump_count = group.get('jump_count', 0)+1
            group['jump_count'] = jump_count
            lr = group['lr']
            for p in group['params']:
                state = self.state[p]
                n_obs = state['step_count']
                
                if n_obs < 2:
                    continue
                
                jump = 1 + jump_mult*lr # a jump is relative to learning rate this is helpful for learning_rate schedules
                n_target = n_obs * jump
                last_n_target = n_target

                # Solve Linear Regression: y = ax + b
                denom = (n_obs * state['sum_x2']) - (state['sum_x']**2)
                if abs(denom) < 1e-9:
                    continue
                
                #a = (n_obs * state['sum_xy'] - state['sum_x'] * state['sum_y']) / denom
                a = (state['sum_xy'].mul(n_obs).sub_(state['sum_y'].mul(state['sum_x']))).div_(denom)
                b = (state['sum_y'].sub_(a.mul(state['sum_x']))) / n_obs
                angle = torch.atan(a) # Teleport!
                state['angle_ema'] = 0.9 * state.get('angle_ema', torch.zeros_like(angle)) + 0.1 * angle
                bias_correction = 1 - 0.9 ** jump_count
                factor = torch.sin(state['angle_ema']/bias_correction)
                #jump = 1 + jump_mult * factor.abs()
                n_target = n_obs * jump
                p.data.copy_(a.mul_(math.log(n_target)).add_(b))
                            


        self._reset_epoch_stats(last_n_target)
        return last_n_target

    def _reset_epoch_stats(self,current_step=0):
        for group in self.param_groups:

            for p in group['params']:
                state = self.state[p]
                state['step_count'] = 0
                state['sum_x'] = 0.0
                state['sum_y'].zero_()
                state['sum_xy'].zero_()
                state['sum_x2'] = 0.0
                # Reset momentum buffer to zero for the next epoch's recovery
                if state['momentum_buffer'] is not None:
                    state['momentum_buffer'].zero_()