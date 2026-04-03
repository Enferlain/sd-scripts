import torch


@torch._dynamo.utils.disable_cache_limit()
@torch.compile()
def stable_spam_clipping_compile_wrapper(
    state: dict,
    grad: torch.Tensor,
    step: int | torch.Tensor,
    scale: float | torch.Tensor = 1.0,
    eps: float | torch.Tensor = 1e-8,
    gamma1: float | torch.Tensor = 0.85,
    gamma2: float | torch.Tensor = 0.99999,
    gamma3: float | torch.Tensor = 0.999,
) -> torch.Tensor:
    return stable_spam_clipping_impl(state, grad, step, scale, eps, gamma1, gamma2, gamma3)


@torch.no_grad()
def stable_spam_clipping_impl(
    state: dict,
    grad: torch.Tensor,
    step: int | torch.Tensor,
    scale: float | torch.Tensor = 1.0,
    eps: float | torch.Tensor = 1e-8,
    gamma1: float | torch.Tensor = 0.85,
    gamma2: float | torch.Tensor = 0.99999,
    gamma3: float | torch.Tensor = 0.999,
) -> torch.Tensor:
    if "ssc_m_norm_t" not in state:
        state["ssc_m_norm_t"] = 0.0
        state["ssc_v_norm_t"] = 0.0
        state["ssc_m_max_t"] = 0.0

    max_grad = torch.max(grad.abs())
    m_max_t = state["ssc_m_max_t"]
    m_max_t = gamma3 * m_max_t + (1 - gamma3) * max_grad
    state["ssc_m_max_t"] = m_max_t

    m_max_hat = m_max_t / (1.0 - gamma3**step)
    grad = torch.where(grad.abs() > m_max_hat, grad / max_grad * m_max_hat, grad)

    grad_norm = torch.norm(grad)
    m_norm_t, v_norm_t = state["ssc_m_norm_t"], state["ssc_v_norm_t"]

    m_norm_t = gamma1 * scale * m_norm_t + (1 - gamma1 * scale) * grad_norm
    v_norm_t = gamma2 * v_norm_t + (1 - gamma2) * grad_norm**2
    m_norm_hat = m_norm_t / (1.0 - (gamma1 * scale) ** step)
    v_norm_hat = v_norm_t / (1.0 - gamma2**step)

    state["ssc_m_norm_t"], state["ssc_v_norm_t"] = m_norm_t, v_norm_t
    c_norm_t = m_norm_hat / (torch.sqrt(v_norm_hat) + eps)
    return torch.where(grad_norm > 0, grad / grad_norm * c_norm_t, grad)


@torch.no_grad()
def stable_spam_clipping_tensors(
    ssc_m_norm_t: torch.Tensor,
    ssc_v_norm_t: torch.Tensor,
    ssc_m_max_t: torch.Tensor,
    grad: torch.Tensor,
    step: int | torch.Tensor,
    scale: float | torch.Tensor = 1.0,
    eps: float | torch.Tensor = 1e-8,
    gamma1: float | torch.Tensor = 0.85,
    gamma2: float | torch.Tensor = 0.99999,
    gamma3: float | torch.Tensor = 0.999,
) -> torch.Tensor:
    max_grad = torch.max(grad.abs())
    m_max_t = gamma3 * ssc_m_max_t + (1 - gamma3) * max_grad
    ssc_m_max_t.copy_(m_max_t)

    m_max_hat = m_max_t / (1.0 - gamma3**step)
    grad = torch.where(grad.abs() > m_max_hat, grad / max_grad * m_max_hat, grad)

    grad_norm = torch.norm(grad)
    m_norm_t = gamma1 * scale * ssc_m_norm_t + (1 - gamma1 * scale) * grad_norm
    v_norm_t = gamma2 * ssc_v_norm_t + (1 - gamma2) * grad_norm**2
    ssc_m_norm_t.copy_(m_norm_t)
    ssc_v_norm_t.copy_(v_norm_t)

    m_norm_hat = m_norm_t / (1.0 - (gamma1 * scale) ** step)
    v_norm_hat = v_norm_t / (1.0 - gamma2**step)
    c_norm_t = m_norm_hat / (torch.sqrt(v_norm_hat) + eps)
    return torch.where(grad_norm > 0, grad / grad_norm * c_norm_t, grad)
