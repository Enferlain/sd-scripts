import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup

from library.optimization.optimizers.utils.math_utils import debias_beta
from library.optimization.optimizers.utils.stochastic import copy_stochastic_


class Alice(BaseOptimizer):
    r"""Adaptive low-dimensional subspace estimation.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
        beta3=0 for Alice-0 optimizer.
    :param alpha: float. scaler.
    :param alpha_c: float. compensation scaler.
    :param update_interval: int. update interval.
    :param rank: int. rank.
    :param gamma: limiter threshold.
    :param leading_basis: int. leading basis.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 0.01,
        betas: Betas = (0.9, 0.9, 0.999),
        alpha: float = 0.25,
        alpha_c: float = 0.2,
        update_interval: int = 10,
        rank: int = 32,
        gamma: float = 1.01,
        leading_basis: int = 10,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        adam_lr: float = 5e-4,
        adam_betas: Betas = (0.9, 0.999),
        adam_weight_decay: float = 0.0,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(alpha, "alpha", 0.0, 1.0)
        self.validate_range(alpha_c, "alpha_c", 0.0, 1.0)
        self.validate_positive(update_interval, "update_interval")
        self.validate_positive(rank, "rank")
        self.validate_positive(gamma, "gamma")
        self.validate_positive(leading_basis, "leading_basis")
        self.validate_non_negative(rank - leading_basis, "rank - leading_basis")
        self.validate_non_negative(weight_decay, "weight_decay")
        self.validate_non_negative(adam_weight_decay, "adam_weight_decay")
        self.validate_non_negative(eps, "eps")
        self.validate_learning_rate(adam_lr)
        self.validate_betas(adam_betas)

        defaults: Defaults = {
            "lr": lr,
            "_lr_ratio": (adam_lr / lr) if lr > 0 else 0,
            "betas": betas,
            "alpha": alpha,
            "alpha_c": alpha_c,
            "update_interval": update_interval,
            "rank": rank,
            "gamma": gamma,
            "leading_basis": leading_basis,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "eps": eps,
            "adam_lr": adam_lr,
            "adam_betas": adam_betas,
            "adam_weight_decay": adam_weight_decay,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "Alice"

    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        pass

    @staticmethod
    def subspace_iteration(
        a: torch.Tensor, mat: torch.Tensor, num_steps: int = 1, jitter: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Perform subspace iteration. Returns eigenvalues, eigenvectors (coeffs), and basis u."""
        # Ensure inputs are at least float32 for stability.
        a_float32 = a.to(torch.float32)
        u = mat.to(torch.float32)

        basis_u = u
        for _ in range(num_steps):
            u, _ = torch.linalg.qr(a_float32 @ u)
            basis_u = u

        # V = basis_u.T @ a @ basis_u has shape (rank, rank).
        v_matrix = basis_u.T @ a_float32 @ basis_u

        # Stabilize eigh via enforced symmetry plus diagonal jitter.
        v_matrix_sym = (v_matrix + v_matrix.T) / 2.0
        rank_dim = v_matrix_sym.shape[0]
        diag_jitter = jitter * torch.eye(rank_dim, device=v_matrix_sym.device, dtype=torch.float32)
        v_matrix_stable = v_matrix_sym + diag_jitter

        vals, vecs = torch.linalg.eigh(v_matrix_stable)

        return vals, vecs, basis_u

    def switch(self, q: torch.Tensor, u_prev: torch.Tensor, rank: int, leading_basis: int) -> torch.Tensor:
        m_dim = q.shape[0]
        original_dtype = q.dtype
        q_float32 = q.to(torch.float32)
        u_prev_float32 = u_prev.to(torch.float32)

        vals, vecs, basis_u = self.subspace_iteration(q_float32, u_prev_float32, num_steps=1, jitter=1e-8)

        eigenvectors_full = basis_u @ vecs

        leading_indices = torch.argsort(vals, descending=True)[:leading_basis]
        u_t1 = eigenvectors_full[:, leading_indices]

        eye_m = torch.eye(m_dim, device=q.device, dtype=torch.float32)
        complement_proj = eye_m - u_t1 @ u_t1.T

        qr_jitter = 1e-8 * torch.randn_like(complement_proj)
        u_c, _ = torch.linalg.qr(complement_proj + qr_jitter)

        num_complement_needed = rank - leading_basis
        if num_complement_needed <= 0:
            u_t2 = torch.empty((m_dim, 0), device=q.device, dtype=torch.float32)
        elif u_c.shape[1] >= num_complement_needed:
            u_t2 = u_c[:, :num_complement_needed]
        else:
            padding = torch.zeros(m_dim, num_complement_needed - u_c.shape[1], device=q.device, dtype=torch.float32)
            u_t2 = torch.cat([u_c[:, : u_c.shape[1]], padding], dim=1)

        final_u = torch.cat([u_t1, u_t2], dim=1)

        if final_u.shape[1] > rank:
            final_u = final_u[:, :rank]
        elif final_u.shape[1] < rank:
            padding = torch.zeros(m_dim, rank - final_u.shape[1], device=q.device, dtype=torch.float32)
            final_u = torch.cat([final_u, padding], dim=1)

        return final_u.to(original_dtype)

    @staticmethod
    def compensation(
        grad: torch.Tensor,
        u: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        gamma: float,
        decay_rate: float,
        rank: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m, n = grad.shape

        sigma = u.T @ grad

        p.mul_(decay_rate).add_(grad.pow(2).sum(dim=0) - sigma.pow(2).sum(dim=0), alpha=1.0 - decay_rate).clamp_min_(1e-8)

        d = torch.zeros_like(grad)
        diag_len: int = min(m, n)
        d[torch.arange(diag_len), torch.arange(diag_len)] = (1.0 / p.sqrt())[:diag_len]

        c_t = math.sqrt(m - rank) * (grad - u @ sigma) * d if m >= rank else torch.zeros_like(grad)

        n = gamma / max(torch.norm(c_t) / phi, gamma) if phi.item() > 0 else torch.ones_like(phi)

        c_t.mul_(n)
        phi = torch.norm(c_t)

        return c_t, phi

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            beta1, beta2, beta3 = group["betas"]
            rank, leading_basis = group["rank"], group["leading_basis"]
            adam_betas = group["adam_betas"]
            beta1_comp = 1 - debias_beta(adam_betas[0], group["step"])
            beta2_hat = debias_beta(adam_betas[1], group["step"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                # Alice doesn't support scalars or dim > 2. Fallback to AdamW.
                if p.ndim == 0 or p.ndim > 2:
                    p_fp32 = p

                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad = grad.to(torch.float32)
                        p_fp32 = p.to(torch.float32)
                        exp_avg, exp_avg_sq = exp_avg.to(torch.float32), exp_avg_sq.to(torch.float32)

                    if group["adam_weight_decay"]:
                        if group["weight_decouple"]:
                            p_fp32.mul_(group["adam_weight_decay"])
                        else:
                            grad.add_(p_fp32, alpha=group["adam_weight_decay"])

                    exp_avg.lerp_(grad, weight=beta1_comp)
                    exp_avg_sq.mul_(beta2_hat).addcmul_(grad, grad, value=1 - beta2_hat)

                    p_fp32.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-group["lr"] * group["_lr_ratio"])

                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)
                    continue

                if len(p.shape) == 1:
                    p = p.unsqueeze(0)  # noqa: PLW2901
                    grad = grad.unsqueeze(0)

                if len(state) == 0:
                    m, n = grad.shape

                    state["u"] = torch.zeros((m, rank), dtype=p.dtype, device=p.device)
                    state["q"] = torch.zeros((rank, rank), dtype=p.dtype, device=p.device)

                    state["m"] = torch.zeros((rank, n), dtype=p.dtype, device=p.device)
                    state["v"] = torch.zeros((rank, n), dtype=p.dtype, device=p.device)

                    state["p"] = torch.zeros((n,), dtype=p.dtype, device=p.device)
                    state["phi"] = torch.zeros((1,), dtype=p.dtype, device=p.device)

                q, u, m, v, phi = state["q"], state["u"], state["m"], state["v"], state["phi"]

                p_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(torch.float32)
                    q, u, m, v, phi = q.to(torch.float32), u.to(torch.float32), m.to(torch.float32), v.to(torch.float32), phi.to(
                        torch.float32
                    )

                self.apply_weight_decay(
                    p=p_fp32,
                    grad=grad,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    weight_decouple=group["weight_decouple"],
                    fixed_decay=False,
                )

                if group["step"] == 1 or group["step"] % group["update_interval"] == 0:
                    q_t = beta3 * (u @ q @ u.T) + (1.0 - beta3) * (grad @ grad.T)
                    u = self.switch(q_t, u, rank, leading_basis)

                sigma = u.T @ grad

                q.mul_(beta3).add_(sigma @ sigma.T, alpha=1.0 - beta3)
                m.mul_(beta1).add_(sigma, alpha=1.0 - beta1)
                v.mul_(beta2).add_(sigma.pow(2), alpha=1.0 - beta2)

                c_t, phi = self.compensation(grad, u, state["p"], state["phi"], group["gamma"], beta1, rank, group["eps"])

                update = u @ (m / (v.sqrt() + group["eps"]))
                update.add_(c_t, alpha=group["alpha_c"])

                p_fp32.add_(update, alpha=-group["lr"] * group["alpha"])

                copy_stochastic_(state["q"], q)
                copy_stochastic_(state["u"], u)
                copy_stochastic_(state["m"], m)
                copy_stochastic_(state["v"], v)
                copy_stochastic_(state["phi"], phi)
                copy_stochastic_(p, p_fp32)

        return loss
