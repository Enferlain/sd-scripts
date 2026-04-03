import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup

from library.optimization.optimizers.utils import NORM_TYPE, adaptive_eps, agc, copy_stochastic_


class ADOPTScheduleFree(BaseOptimizer):
    r"""Schedule-Free ADOPT.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.9, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay at y, i.e. a L2 penalty (default: 0.0).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: False)
        stable_weight_decay (bool): 
            Requires weight_decouple be True. Applies stable weight decay - https://arxiv.org/abs/2011.11152 (default: False)
        adaptive_clip (float):
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        debias_beta2 (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: False)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        debias_beta2: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'debias_beta2':debias_beta2,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPTScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['exp_avg_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['exp_avg_mean_sqrt'] = 0.0

            param_size: int = 0
            exp_avg_sq_sum: float = 0.0

            beta1, beta2 = group['betas']

            if group["debias_beta2"]:
                bias_correction2: float = self.debias(beta2, group['step'])
            else:
                bias_correction2 = 1.0

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            group["eps"]
            group["eps2"]
            group["eps_floor"]

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                p_fp32 = p
                state = self.state[p]

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    param_size += p.numel()                

                if len(state) == 0:
                    state['z'] = p.clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)

                z, exp_avg_sq = state['z'], state['exp_avg_sq']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, exp_avg_sq = z.to(torch.float32), exp_avg_sq.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    exp_avg_sq.addcmul_(grad, grad.conj())
                else:
                    de_nom = exp_avg_sq.div(bias_correction2).sqrt_().add_(curr_eps)

                    update = grad.div(de_nom)
                    update.clamp_(-adopt_clip, adopt_clip)

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        update.add_(p_fp32, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(update, alpha=adaptive_y_lr)

                    z.sub_(update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    exp_avg_sq_sum += exp_avg_sq.div(bias_correction2).sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['exp_avg_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

        return loss
    
class ADOPTEMAMixScheduleFree(BaseOptimizer):
    r"""Schedule-Free ADOPT + AdEMAMix slow ema.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum, exponential moving average squared, and slow ema/momentum (default: 0.9, 0.9999, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay at y, i.e. a L2 penalty (default: 0.0).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: False)
        stable_weight_decay (bool): 
            Requires weight_decouple be True. Applies stable weight decay - https://arxiv.org/abs/2011.11152 (default: False)
        adaptive_clip (float):
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        alpha (float): 
            usually between 2 and 5 would work well. (default: 2)
        t_alpha_beta3 (Optional[float]): 
            Steps to warmup alpha and beta 3. Total number of steps is recommended when needed. (Default: None)
        cautious (bool):
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: True)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        cautious: bool = True,
        alpha: float = 2.0,
        t_alpha_beta3: float | None = None,
        debias_beta2: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'alpha': alpha,
            't_alpha_beta3': t_alpha_beta3,
            'cautious': cautious,
            'debias_beta2':debias_beta2,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPTEMAMixScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['exp_avg_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_slow'] = torch.zeros_like(p)

    @staticmethod
    def schedule_alpha(t_alpha_beta3: float | None, step: int, alpha: float) -> float:
        if t_alpha_beta3 is None:
            return alpha
        return min(step * alpha / t_alpha_beta3, alpha)

    @staticmethod
    def schedule_beta3(t_alpha_beta3: float | None, step: int, beta1: float, beta3: float, eps: float) -> float:
        if t_alpha_beta3 is None:
            return beta3

        # Add eps to prevent log 0
        log_beta1, log_beta3 = math.log(beta1 + eps), math.log(beta3)

        return min(
            math.exp(
                log_beta1 * log_beta3 / ((1.0 - step / t_alpha_beta3) * log_beta3 + (step / t_alpha_beta3) * log_beta1)
            ),
            beta3,
        )

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['exp_avg_mean_sqrt'] = 0.0

            param_size: int = 0
            exp_avg_sq_sum: float = 0.0

            beta1, beta2, beta3 = group['betas']

            if group["debias_beta2"]:
                bias_correction2: float = self.debias(beta2, group['step'])
            else:
                bias_correction2 = 1.0

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            group["eps"]
            group["eps2"]
            group["eps_floor"]


            alpha_t: float = self.schedule_alpha(group['t_alpha_beta3'], group['step'], group['alpha'])
            beta3_t: float = self.schedule_beta3(group['t_alpha_beta3'], group['step'], beta1, beta3, 1e-8)


            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                p_fp32 = p
                state = self.state[p]

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    param_size += p.numel()                

                if len(state) == 0:
                    state['z'] = p.clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_slow'] = torch.zeros_like(p)

                z, exp_avg_sq, exp_avg_slow = state['z'], state['exp_avg_sq'], state['exp_avg_slow']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, exp_avg_sq, exp_avg_slow = z.to(torch.float32), exp_avg_sq.to(torch.float32), exp_avg_slow.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    exp_avg_sq.addcmul_(grad, grad.conj())
                else:
                    de_nom = exp_avg_sq.div(bias_correction2).sqrt_().add_(curr_eps)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                    exp_avg_slow.mul_(beta3_t).add_(grad, alpha=1.0 - beta3_t)
                    slow_ema_update = (alpha_t * exp_avg_slow).div(de_nom)
                    slow_ema_update.clamp_(-adopt_clip, adopt_clip)

                    grad_update = grad.div(de_nom)
                    grad_update.clamp_(-adopt_clip, adopt_clip)

                    if group["cautious"]:
                        # compute norm gradient
                        mask = (slow_ema_update * grad_update > 0).to(grad.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        slow_ema_update.mul_(mask)

                    full_update = grad_update + slow_ema_update

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        full_update.add_(p_fp32, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(full_update, alpha=adaptive_y_lr)

                    z.sub_(full_update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    exp_avg_sq_sum += exp_avg_sq.div(bias_correction2).sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(state["exp_avg_slow"], exp_avg_slow)
                    copy_stochastic_(p, p_fp32)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['exp_avg_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

        return loss
    
class ADOPTNesterovScheduleFree(BaseOptimizer):
    r"""Schedule-Free ADOPT + Adan style nesterov momentum.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum, grad diff ema, and exponential moving average squared (default: 0.9, 0.92, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay at y, i.e. a L2 penalty (default: 0.0).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: False)
        stable_weight_decay (bool): 
            Requires weight_decouple be True. Applies stable weight decay - https://arxiv.org/abs/2011.11152 (default: False)
        adaptive_clip (float):
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        cautious (bool):
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: True)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.92, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        cautious: bool = True,
        debias_beta2: bool = False,
        debias_beta3: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'debias_beta2':debias_beta2,
            'debias_beta3':debias_beta3,
            'cautious': cautious,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPTNesterovScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['exp_avg_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_diff'] = torch.zeros_like(p)
                state['previous_grad'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['exp_avg_mean_sqrt'] = 0.0

            param_size: int = 0
            exp_avg_sq_sum: float = 0.0

            beta1, beta2, beta3 = group['betas']

            if group["debias_beta2"]:
                self.debias(beta2, group['step'])
            else:
                pass

            if group["debias_beta3"]:
                bias_correction3: float = self.debias(beta3, group['step'])
            else:
                bias_correction3 = 1.0

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            group["eps"]
            group["eps2"]
            group["eps_floor"]

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                p_fp32 = p
                state = self.state[p]

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    param_size += p.numel()                

                if len(state) == 0:
                    state['z'] = p.clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['previous_grad'] = -p.grad.to(dtype=p.dtype, copy=True).detach()

                z, exp_avg_sq, exp_avg_diff, grad_diff = state['z'], state['exp_avg_sq'], state['exp_avg_diff'], state['previous_grad']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, exp_avg_sq, exp_avg_diff, grad_diff = z.to(torch.float32), exp_avg_sq.to(torch.float32), exp_avg_diff.to(torch.float32), grad_diff.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                grad_diff.add_(grad)

                if group['step'] == 1:
                    grad_diff.mul_(beta2).add_(grad)
                    exp_avg_sq.addcmul_(grad_diff, grad_diff.conj())
                else:
                    de_nom = exp_avg_sq.div(bias_correction3).sqrt_().add_(curr_eps)
                    exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=1.0 - beta2)

                    grad_diff.mul_(beta2).add_(grad)
                    exp_avg_sq.mul_(beta3).addcmul_(grad_diff, grad_diff.conj(), value=1 - beta3)
                    
                    ema_diff_update = exp_avg_diff.div(de_nom)
                    ema_diff_update.clamp_(-adopt_clip, adopt_clip)

                    grad_update = grad.div(de_nom)
                    grad_update.clamp_(-adopt_clip, adopt_clip)

                    if group["cautious"]:
                        # compute norm gradient
                        mask = (ema_diff_update * grad_update > 0).to(grad_update.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        ema_diff_update.mul_(mask)

                    full_update = grad_update + ema_diff_update

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        full_update.add_(p_fp32, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(full_update, alpha=adaptive_y_lr)

                    z.sub_(full_update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    exp_avg_sq_sum += exp_avg_sq.div(bias_correction3).sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(state['exp_avg_diff'], exp_avg_diff)
                    copy_stochastic_(state['previous_grad'], -grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    state['previous_grad'].copy_(-grad)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['exp_avg_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

        return loss

class ADOPTMARSScheduleFree(BaseOptimizer):
    r"""Schedule-Free ADOPT + MARS Correction.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.9, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay at y, i.e. a L2 penalty (default: 0.0).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: False)
        stable_weight_decay (bool): 
            Requires weight_decouple be True. Applies stable weight decay - https://arxiv.org/abs/2011.11152 (default: False)
        adaptive_clip (float):
            Adaptive clip value to apply to the MARS corrected gradient - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        gamma (float):
            Scaling value for the MARS style correction of the gradient, 0.025 or 0.05 are recommended by the paper, 
            larger values apply more correction, and will require higher LRs to offset. (default: 0.025)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        gamma: float = 0.025,
        debias_beta2: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'gamma':gamma,
            'debias_beta2':debias_beta2,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPTMARSScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['exp_avg_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['previous_grad'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['exp_avg_mean_sqrt'] = 0.0

            param_size: int = 0
            exp_avg_sq_sum: float = 0.0

            beta1, beta2 = group['betas']

            if group["debias_beta2"]:
                bias_correction2: float = self.debias(beta2, group['step'])
            else:
                bias_correction2 = 1.0

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            group["eps"]
            group["eps2"]
            group["eps_floor"]
            gamma = group["gamma"]

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                p_fp32 = p
                state = self.state[p]

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    param_size += p.numel()                

                if len(state) == 0:
                    state['z'] = p.clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['previous_grad'] = p.grad.to(dtype=p.dtype, copy=True).detach()

                z, exp_avg_sq = state['z'], state['exp_avg_sq']
                previous_grad = state['previous_grad']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, exp_avg_sq = z.to(torch.float32), exp_avg_sq.to(torch.float32)
                    previous_grad = previous_grad.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                # MARS Calculate cₜ (gradient with correction term)
                c_t = (grad - previous_grad).mul_(gamma * (beta1 / (1.0 - beta1))).add_(grad)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    c_t = agc(p=p_fp32, grad=c_t, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)
                if group['step'] == 1:
                    exp_avg_sq.addcmul_(c_t, c_t.conj())
                else:
                    de_nom = exp_avg_sq.div(bias_correction2).sqrt_().add_(curr_eps)
                    exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t.conj(), value=1.0 - beta2)

                    grad_update = c_t.div(de_nom)
                    grad_update.clamp_(-adopt_clip, adopt_clip)

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        grad_update.add_(p_fp32, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(grad_update, alpha=adaptive_y_lr)

                    z.sub_(grad_update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    exp_avg_sq_sum += exp_avg_sq.div(bias_correction2).sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state['z'], z)
                    copy_stochastic_(state['exp_avg_sq'], exp_avg_sq)
                    copy_stochastic_(state['previous_grad'], grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    state['previous_grad'].copy_(grad)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['exp_avg_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

        return loss
    
class FADOPTScheduleFree(BaseOptimizer):
    r"""Schedule-Free fisher ADOPT.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.9, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay at y, i.e. a L2 penalty (default: 0.0).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: False)
        stable_weight_decay (bool): 
            Requires weight_decouple be True. Applies stable weight decay - https://arxiv.org/abs/2011.11152 (default: False)
        adaptive_clip (float):
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        fisher_clip (float):
            Required clipping fisher applies to the natual gradient and natural weights. (default: 1.0)
            
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        fisher_clip: float = 1.0,
        debias_beta2: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'train_mode': True,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'fisher_clip':fisher_clip,
            'debias_beta2':debias_beta2,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FADOPTScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['fim_mean_sqrt'] = 0.0

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2 = group['betas']

            if group["debias_beta2"]:
                current_beta2: float = self.debias_beta(beta2, group['step'])
            else:
                current_beta2 = beta2

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            group["eps"]
            group["eps2"]
            group["eps_floor"]
            fisher_clip = group["fisher_clip"]

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                p_fp32 = p
                state = self.state[p]

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    param_size += p.numel()                

                if len(state) == 0:
                    state['z'] = p.clone()
                    state['fim'] = torch.ones_like(p)

                z, fim = state['z'], state['fim']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, fim = z.to(torch.float32), fim.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    fim.addcmul_(grad, grad.conj()).clamp_(-adopt_clip, adopt_clip)
                else:
                    fim_base = fim.sqrt().add_(curr_eps)
                    fim.mul_(current_beta2).addcmul_(grad, grad.conj(), value=1 - current_beta2).clamp_(-adopt_clip, adopt_clip)

                    grad_nat = grad.div(fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(fisher_clip, rms) / fisher_clip
                    grad_nat.div_(divisor)

                    update = grad_nat
                    
                    # Perform weight decay
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['fim_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['fim_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        grad_weights = p_fp32.div(fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(fisher_clip, rms) / fisher_clip
                        grad_weights.div_(divisor)

                        update.add_(grad_weights, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(update, alpha=adaptive_y_lr)

                    z.sub_(update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    fim_sum += fim.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(state['fim'], fim)
                    copy_stochastic_(p, p_fp32)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

        return loss

class FADOPTEMAMixScheduleFree(BaseOptimizer):
    r"""Schedule-Free fisher ADOPT + AdEMAMix slow ema.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum, exponential moving average squared, and slow ema/momentum (default: 0.9, 0.9999, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay at y, i.e. a L2 penalty (default: 0.0).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: False)
        stable_weight_decay (bool): 
            Requires weight_decouple be True. Applies stable weight decay - https://arxiv.org/abs/2011.11152 (default: False)
        adaptive_clip (float):
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        alpha (float): 
            usually between 2 and 5 would work well. (default: 2)
        t_alpha_beta3 (Optional[float]): 
            Steps to warmup alpha and beta 3. Total number of steps is recommended when needed. (Default: None)
        cautious (bool):
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: True)
        fisher_clip (float):
            Required clipping fisher applies to the natual gradient and natural weights. (default: 1.0)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        fisher_clip: float = 1.0,
        cautious: bool = True,
        alpha: float = 2.0,
        t_alpha_beta3: float | None = None,
        debias_beta2: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'train_mode': True,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'fisher_clip':fisher_clip,
            'cautious':cautious,
            'alpha':alpha,
            't_alpha_beta3':t_alpha_beta3,
            'debias_beta2':debias_beta2,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FADOPTEMAMixScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['fim'] = torch.ones_like(p)
                state['exp_avg_slow'] = torch.zeros_like(p)

    @staticmethod
    def schedule_alpha(t_alpha_beta3: float | None, step: int, alpha: float) -> float:
        if t_alpha_beta3 is None:
            return alpha
        return min(step * alpha / t_alpha_beta3, alpha)

    @staticmethod
    def schedule_beta3(t_alpha_beta3: float | None, step: int, beta1: float, beta3: float, eps: float) -> float:
        if t_alpha_beta3 is None:
            return beta3

        # Add eps to prevent log 0
        log_beta1, log_beta3 = math.log(beta1 + eps), math.log(beta3)

        return min(
            math.exp(
                log_beta1 * log_beta3 / ((1.0 - step / t_alpha_beta3) * log_beta3 + (step / t_alpha_beta3) * log_beta1)
            ),
            beta3,
        )

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['fim_mean_sqrt'] = 0.0

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2, beta3 = group['betas']

            if group["debias_beta2"]:
                current_beta2: float = self.debias_beta(beta2, group['step'])
            else:
                current_beta2 = beta2

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            group["eps"]
            group["eps2"]
            group["eps_floor"]
            fisher_clip = group["fisher_clip"]

            alpha_t: float = self.schedule_alpha(group['t_alpha_beta3'], group['step'], group['alpha'])
            beta3_t: float = self.schedule_beta3(group['t_alpha_beta3'], group['step'], beta1, beta3, 1e-8)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                p_fp32 = p
                state = self.state[p]

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    param_size += p.numel()                

                if len(state) == 0:
                    state['z'] = p.clone()
                    state['fim'] = torch.ones_like(p)
                    state['exp_avg_slow'] = torch.ones_like(p)

                z, fim, exp_avg_slow = state['z'], state['fim'], state['exp_avg_slow']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, fim, exp_avg_slow = z.to(torch.float32), fim.to(torch.float32), exp_avg_slow.to(torch.float32),
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    fim.addcmul_(grad, grad.conj()).clamp_(-adopt_clip, adopt_clip)
                else:
                    fim_base = fim.sqrt().add_(curr_eps)
                    fim.mul_(current_beta2).addcmul_(grad, grad.conj(), value=1 - current_beta2).clamp_(-adopt_clip, adopt_clip)

                    grad_nat = grad.div(fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(fisher_clip, rms) / fisher_clip
                    grad_nat.div_(divisor)

                    exp_avg_slow.mul_(beta3_t).add_(grad_nat, alpha=1.0 - beta3_t)
                    slow_ema_update = (alpha_t * exp_avg_slow)

                    if group["cautious"]:
                        # compute norm gradient
                        mask = (slow_ema_update * grad_nat > 0).to(grad_nat.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        slow_ema_update.mul_(mask)

                    update = grad_nat + slow_ema_update
                    
                    # Perform weight decay
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['fim_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['fim_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        grad_weights = p_fp32.div(fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(fisher_clip, rms) / fisher_clip
                        grad_weights.div_(divisor)

                        update.add_(grad_weights, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(update, alpha=adaptive_y_lr)

                    z.sub_(update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    fim_sum += fim.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(state['fim'], fim)
                    copy_stochastic_(state["exp_avg_slow"], exp_avg_slow)
                    copy_stochastic_(p, p_fp32)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

        return loss
    
class FADOPTNesterovScheduleFree(BaseOptimizer):
    r"""Schedule-Free fisher ADOPT.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum, grad diff ema, and exponential moving average squared (default: 0.9, 0.92, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay at y, i.e. a L2 penalty (default: 0.0).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: False)
        stable_weight_decay (bool): 
            Requires weight_decouple be True. Applies stable weight decay - https://arxiv.org/abs/2011.11152 (default: False)
        adaptive_clip (float):
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        fisher_clip (float):
            Required clipping fisher applies to the natual gradient and natural weights. (default: 1.0)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.92, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        fisher_clip: float = 1.0,
        cautious: bool = True,
        debias_beta2: bool = False,
        debias_beta3: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'fisher_clip':fisher_clip,
            'cautious':cautious,
            'debias_beta2':debias_beta2,
            'debias_beta3':debias_beta3,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FADOPTNesterovScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['fim'] = torch.ones_like(p)
                state['exp_avg_diff'] = torch.zeros_like(p)
                state['previous_grad'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['fim_mean_sqrt'] = 0.0

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2, beta3 = group['betas']

            if group["debias_beta2"]:
                current_beta2: float = self.debias_beta(beta2, group['step'])
            else:
                current_beta2 = beta2

            if group["debias_beta3"]:
                current_beta3: float = self.debias_beta(beta3, group['step'])
            else:
                current_beta3 = beta3

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            group["eps"]
            group["eps2"]
            group["eps_floor"]
            fisher_clip = group["fisher_clip"]

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                p_fp32 = p
                state = self.state[p]

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    param_size += p.numel()                

                if len(state) == 0:
                    state['z'] = p.clone()
                    state['fim'] = torch.ones_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['previous_grad'] = -p.grad.to(dtype=p.dtype, copy=True).detach()

                z, fim, exp_avg_diff, grad_diff = state['z'], state['fim'], state['exp_avg_diff'], state['previous_grad']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, fim, exp_avg_diff, grad_diff = z.to(torch.float32), fim.to(torch.float32), exp_avg_diff.to(torch.float32), grad_diff.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                grad_diff.add_(grad)

                if group['step'] == 1:
                    grad_diff.mul_(current_beta2).add_(grad)
                    fim.addcmul_(grad_diff, grad_diff.conj()).clamp_(-adopt_clip, adopt_clip)
                else:
                    fim_base = fim.sqrt().add_(curr_eps)
                    exp_avg_diff.mul_(current_beta2).add_(grad_diff, alpha=1.0 - current_beta2)

                    grad_diff.mul_(current_beta2).add_(grad)
                    fim.mul_(current_beta3).addcmul_(grad_diff, grad_diff.conj(), value=1 - current_beta3).clamp_(-adopt_clip, adopt_clip)

                    grad_nat = grad.div(fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(fisher_clip, rms) / fisher_clip
                    grad_nat.div_(divisor)

                    ema_diff_update = exp_avg_diff

                    if group["cautious"]:
                        # compute norm gradient
                        mask = (ema_diff_update * grad_nat > 0).to(grad_nat.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        ema_diff_update.mul_(mask)

                    update = grad_nat + ema_diff_update
                    
                    # Perform weight decay
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['fim_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['fim_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - group['weight_decay'] * lr * swd_scaling)
                    elif group["weight_decay"] != 0:
                        grad_weights = p_fp32.div(fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(fisher_clip, rms) / fisher_clip
                        grad_weights.div_(divisor)

                        update.add_(grad_weights, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(update, alpha=adaptive_y_lr)

                    z.sub_(update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    fim_sum += fim.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(state['fim'], fim)
                    copy_stochastic_(state['exp_avg_diff'], exp_avg_diff)
                    copy_stochastic_(state['previous_grad'], -grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    state['previous_grad'].copy_(-grad)

            if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

        return loss
    
class FADOPTMARSScheduleFree(BaseOptimizer):
    r"""Schedule-Free fisher ADOPT + MARS Correction..
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.9, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay at y, i.e. a L2 penalty (default: 0.0).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: False)
        stable_weight_decay (bool): 
            Requires weight_decouple be True. Applies stable weight decay - https://arxiv.org/abs/2011.11152 (default: False)
        adaptive_clip (float):
            Adaptive clip value to apply to the MARS corrected gradient - https://arxiv.org/abs/2102.06171 (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updates due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        r (float): 
            use polynomial weighting in the average with power r.  (Default: 0.0)
        weight_lr_power (float): 
            during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting. (Default: 2,0)
        gamma (float):
            Scaling value for the MARS style correction of the gradient, 0.025 or 0.05 are recommended by the paper, 
            larger values apply more correction, and will require higher LRs to offset. (default: 0.025)
        fisher_clip (float):
            Required clipping fisher applies to the natual gradient and natural weights. (default: 1.0)
        debias_beta2 (bool):
            Apply bias correction to denominator of updates (adaptive LR). (Default: False)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        weight_decay_lr_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        fisher_clip: float = 1.0,
        gamma: float = 0.025,
        debias_beta2: bool = False,
        weight_decay_lr_max: float | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.train_mode = False

        if weight_decay_lr_decouple:
            self.validate_non_negative(weight_decay_lr_max, 'weight_decay_lr_max')

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'fisher_clip':fisher_clip,
            'gamma': gamma,
            'debias_beta2':debias_beta2,
            'weight_decay_lr_decouple':weight_decay_lr_decouple,
            'weight_decay_lr_max':weight_decay_lr_max,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FADOPTMARSScheduleFree'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if not self.train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p_fp32 = p

                        z = state['z']

                        # unpack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            z = z.to(torch.float32)
                            p_fp32 = p.to(dtype=torch.float32, copy=True)

                        p_fp32.data.lerp_(end=z, weight=1.0 - beta1)

                        # pack
                        if p.dtype in {torch.float16, torch.bfloat16}:
                            copy_stochastic_(p, p_fp32)
                self.train_mode = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
                state['fim'] = torch.ones_like(p)
                state['previous_grad'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                group['fim_mean_sqrt'] = 0.0

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2 = group['betas']

            if group["debias_beta2"]:
                current_beta2: float = self.debias_beta(beta2, group['step'])
            else:
                current_beta2 = beta2

            lr: float = group['lr']

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            group["eps"]
            group["eps2"]
            group["eps_floor"]
            fisher_clip = group["fisher_clip"]
            gamma = group["gamma"]
            weight_decay_lr_decouple = group["weight_decay_lr_decouple"]
            weight_decay_lr_max = group["weight_decay_lr_max"]
            weight_decay = group["weight_decay"]
            weight_decouple = group['weight_decouple']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                p_fp32 = p
                state = self.state[p]

                if weight_decay != 0 and weight_decouple and group['stable_weight_decay']:
                    param_size += p.numel()                

                if len(state) == 0:
                    state['z'] = p.clone()
                    state['fim'] = torch.ones_like(p)
                    state['previous_grad'] = p.grad.to(dtype=p.dtype, copy=True).detach()

                z, fim, previous_grad = state['z'], state['fim'], state['previous_grad']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, fim, previous_grad = z.to(torch.float32), fim.to(torch.float32), previous_grad.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)
            
                # MARS Calculate cₜ (gradient with correction term)
                c_t = (grad - previous_grad).mul_(gamma * (beta1 / (1.0 - beta1))).add_(grad)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    c_t = agc(p=p_fp32, grad=c_t, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                curr_eps = adaptive_eps(grad, group)

                if group['step'] == 1:
                    fim.addcmul_(c_t, c_t.conj()).clamp_(-adopt_clip, adopt_clip)
                else:
                    fim_base = fim.sqrt().add_(curr_eps)
                    fim.mul_(current_beta2).addcmul_(c_t, c_t.conj(), value=1 - current_beta2).clamp_(-adopt_clip, adopt_clip)

                    grad_nat = c_t.div(fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(fisher_clip, rms) / fisher_clip
                    grad_nat.div_(divisor)
                    
                    # Perform weight decay
                    if weight_decay != 0 and weight_decouple:
                        if group['stable_weight_decay'] and group['fim_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['fim_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.mul_(1.0 - weight_decay * (lr if not weight_decay_lr_decouple else (lr / weight_decay_lr_max)) * swd_scaling)
                    elif weight_decay != 0:
                        grad_weights = p_fp32.div(fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(fisher_clip, rms) / fisher_clip
                        grad_weights.div_(divisor)

                        grad_nat.add_(grad_weights, alpha=weight_decay)

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(grad_nat, alpha=adaptive_y_lr)

                    z.sub_(grad_nat, alpha=lr)

                if weight_decay != 0 and weight_decouple and group['stable_weight_decay']:
                    fim_sum += fim.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state['z'], z)
                    copy_stochastic_(state['fim'], fim)
                    copy_stochastic_(state['previous_grad'], grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    state['previous_grad'].copy_(grad)

            if weight_decay != 0 and weight_decouple and group['stable_weight_decay']:
                group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

        return loss
