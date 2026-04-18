from __future__ import annotations

from library.adapters import oft as legacy_oft
from library.adapters.runtime import AdapterBuildRequest


def create_adapter(request: AdapterBuildRequest):
    """Repo-owned wrapper for the built-in OFT adapter runtime."""

    settings = dict(request.adapter.settings)
    adapter_rank = settings.pop("adapter_rank", None)
    adapter_alpha = settings.pop("adapter_alpha", None)
    neuron_dropout = settings.pop("neuron_dropout", None)

    if request.context.model.denoiser is None:
        raise ValueError("OFT adapter runtime requires a denoiser module")

    adapter = legacy_oft.create_adapter(
        request.context.multiplier,
        adapter_rank,
        adapter_alpha,
        request.context.model.vae,
        request.context.model.text_encoder,
        request.context.model.denoiser,
        neuron_dropout=neuron_dropout,
        **settings,
    )
    adapter.adapter_resolved_targets = request.resolved_targets
    return adapter


def create_adapter_from_weights(request: AdapterBuildRequest, weights_path: str):
    """Repo-owned wrapper for loading the built-in OFT adapter from weights."""

    settings = dict(request.adapter.settings)
    settings.pop("adapter_rank", None)
    settings.pop("adapter_alpha", None)
    settings.pop("neuron_dropout", None)

    if request.context.model.denoiser is None:
        raise ValueError("OFT adapter runtime requires a denoiser module")

    adapter, weights_sd = legacy_oft.create_adapter_from_weights(
        request.context.multiplier,
        weights_path,
        request.context.model.vae,
        request.context.model.text_encoder,
        request.context.model.denoiser,
        for_inference=request.context.for_inference,
        **settings,
    )
    adapter.adapter_resolved_targets = request.resolved_targets
    return adapter, weights_sd
