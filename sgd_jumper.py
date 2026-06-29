import torch
from torch.optim import Optimizer
import math

def triangle_wave(t, T=1.0, max=1.0 , min=0.1):
    x = t % T
    if x < T/2:
        return min + (max - min) * (2 * x / T)
    else:
        return min + (max - min) * (2 - 2 * x / T)
activations = {
    'linear': lambda x: x,
    'log': lambda x: math.log(x + 1.0),
    'sqrt': lambda x: math.sqrt(x)
}
# lr_SWR is the Standing Wave Ratio for the learning rate lr_max/lr_min, used to determine lr_min in the traingle wave function, the reason for using SWR instead of lr_min directly is to allow for global schedulers to be used with the optimizer, and lr_min can be adjusted accordingly. The default value of 1e-2 is a reasonable starting point for many tasks, but it can be tuned based on the specific problem and dataset.
class Jumper(Optimizer):
    def __init__(self, params,steps_per_epoch, lr=0.5, lr_SWR=1e-2, momentum=0.0, gamma=1.5,fit_type='log',weight_decay=0.0):
        """
        SGD-Jumper: Trend extrapolation of weights.
        
        The optimizer fits a linear or any a.f(x) + b trend to weight updates within an epoch 
        and 'jumps' forward at the end of the epoch.
        """
        print(f"Initializing Jumper with steps_per_epoch={steps_per_epoch}, lr={lr}, lr_SWR={lr_SWR}, momentum={momentum}, gamma={gamma}, weight_decay={weight_decay}")
        if fit_type in activations:
            self.activation = activations[fit_type]
        else:
            raise ValueError(f"Invalid fit_type: {fit_type}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
    
        defaults = dict(lr=lr, lr_SWR=lr_SWR, momentum=momentum, gamma=gamma, steps_per_epoch=steps_per_epoch, weight_decay=weight_decay)
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
            lr_SWR = group['lr_SWR']
            lr_min = lr / lr_SWR
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
                bias_correction = 1 - momentum ** (state['step_count'] + 1)
                lr_effective = triangle_wave(state['step_count'], steps_per_epoch, lr, lr_min)
                p.data.add_(d_p, alpha=-lr_effective * bias_correction)

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr_effective)

                # 3. Accumulate Linear Regression Stats
                x =  self.activation((state['step_count']))

                state['sum_x'] += x
                state['sum_y'].add_(p)
                state['sum_xy'].add_(p.mul(x))
                state['sum_x2'] += x**2
                state['step_count'] += 1

        return loss

    @torch.no_grad()
    def jump(self):
        """Fits ax+b to weight history and projects to n = steps * gamma"""
        last_n_target = 0
        for group in self.param_groups:
            gamma = group['gamma']
            jump_count = group.get('jump_count', 0)+1
            group['jump_count'] = jump_count
            lr = group['lr']
            for p in group['params']:
                state = self.state[p]
                n_obs = state['step_count']
                
                if n_obs < 2:
                    continue
                
                jump = 1 + gamma*lr # a jump is relative to learning rate this is helpful for learning_rate schedules
                n_target = n_obs * jump
                last_n_target = n_target

                # Solve Linear Regression: y = ax + b
                denom = (n_obs * state['sum_x2']) - (state['sum_x']**2)
                if abs(denom) < 1e-9:
                    continue
                
                a = (state['sum_xy'].mul(n_obs).sub_(state['sum_y'].mul(state['sum_x']))).div_(denom)
                b = (state['sum_y'].sub_(a.mul(state['sum_x']))) / n_obs
                p.data.copy_(a.mul_(self.activation(n_target)).add_(b))
                            


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