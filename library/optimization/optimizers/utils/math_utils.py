import math


def debias_beta(beta: float, step: int) -> float:
    """Apply the Adam-style debias correction into beta."""
    return (beta**step - beta) / (beta**step - 1)


def schedule_beta_tc(t_beta: float | None, step: int, beta_initial: float, beta_final: float, eps: float = 1e-8) -> float:
    """Warm beta toward a target value with the donor time-constant schedule."""
    if t_beta is None:
        return beta_initial

    log_beta_initial = math.log(max(beta_initial, eps))
    log_beta_final = math.log(beta_final)
    return min(
        math.exp(
            log_beta_initial * log_beta_final
            / ((1.0 - step / t_beta) * log_beta_final + (step / t_beta) * log_beta_initial)
        ),
        beta_final,
    )
