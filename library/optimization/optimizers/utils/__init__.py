from library.optimization.optimizers.utils.adagc import adagc_global_clipping_calc, apply_adagc_clipping_and_update_gamma
from library.optimization.optimizers.utils.clipping import NORM_TYPE, adaptive_eps, agc
from library.optimization.optimizers.utils.frequency import filter_grad
from library.optimization.optimizers.utils.math_utils import debias_beta, schedule_beta_tc
from library.optimization.optimizers.utils.norms import get_global_gradient_norm
from library.optimization.optimizers.utils.orthograd import (
    bias_rms,
    bias_rms_compile,
    orthograd_atan,
    paper_orthograd,
    paper_orthograd_compile,
    zero_power_via_newton_schulz_6,
    zero_power_via_newton_schulz_6_compile,
)
from library.optimization.optimizers.utils.schedules import CosineDecay, SSCCosineDecay
from library.optimization.optimizers.utils.second_moment import create_factored_dims, get_denom, update_second_moment
from library.optimization.optimizers.utils.spam import spam_grad_clipping, spam_grad_clipping_logging
from library.optimization.optimizers.utils.stable_spam import (
    stable_spam_clipping_compile_wrapper,
    stable_spam_clipping_impl,
    stable_spam_clipping_tensors,
)
from library.optimization.optimizers.utils.state import resolve_state_storage_dtype
from library.optimization.optimizers.utils.stochastic import copy_stochastic_
from library.optimization.optimizers.utils.types import CLIP_TYPE, STATE_PRECISION, UPDATE_STRATEGY
from library.optimization.optimizers.utils.update import apply_update_strategies


__all__ = [
    "CLIP_TYPE",
    "CosineDecay",
    "NORM_TYPE",
    "STATE_PRECISION",
    "SSCCosineDecay",
    "UPDATE_STRATEGY",
    "adagc_global_clipping_calc",
    "adaptive_eps",
    "agc",
    "apply_adagc_clipping_and_update_gamma",
    "apply_update_strategies",
    "bias_rms",
    "bias_rms_compile",
    "copy_stochastic_",
    "create_factored_dims",
    "debias_beta",
    "filter_grad",
    "get_denom",
    "get_global_gradient_norm",
    "orthograd_atan",
    "paper_orthograd",
    "paper_orthograd_compile",
    "resolve_state_storage_dtype",
    "schedule_beta_tc",
    "spam_grad_clipping",
    "spam_grad_clipping_logging",
    "stable_spam_clipping_tensors",
    "stable_spam_clipping_compile_wrapper",
    "stable_spam_clipping_impl",
    "update_second_moment",
    "zero_power_via_newton_schulz_6",
    "zero_power_via_newton_schulz_6_compile",
]
