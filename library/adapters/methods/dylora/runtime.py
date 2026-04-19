from __future__ import annotations

from library.adapters import dylora as legacy_dylora
from library.adapters.runtime import AdapterBuildRequest
from library.adapters.shared import AdapterTrainableParameterRef, attach_trainable_parameter_provider


def _resolve_component_labels(resolved_targets) -> dict[str, str]:
    labels: dict[str, str] = {}
    for target in resolved_targets.targets:
        component_key = target.metadata.get("component_key")
        if isinstance(component_key, str):
            labels[component_key] = target.component
    return labels


def _attach_trainable_ref_provider(adapter, request: AdapterBuildRequest):
    def describe_trainable_parameter_refs() -> list[AdapterTrainableParameterRef]:
        component_labels = _resolve_component_labels(adapter.adapter_resolved_targets)
        text_encoder_keys = [key for key in sorted(component_labels) if key.startswith("text_encoder")]
        refs: list[AdapterTrainableParameterRef] = []

        for lora in list(getattr(adapter, "text_encoder_loras", [])) + list(getattr(adapter, "unet_loras", [])):
            if lora.lora_name.startswith(legacy_dylora.DyLoRAAdapter.LORA_PREFIX_UNET):
                component_key = "denoiser"
            else:
                if not text_encoder_keys:
                    raise ValueError("DyLoRA trainable refs require at least one resolved text encoder target")
                component_key = text_encoder_keys[0]
            component_label = component_labels.get(component_key, component_key)

            for param_name, param in lora.named_parameters():
                refs.append(
                    AdapterTrainableParameterRef(
                        param=param,
                        name=f"{lora.lora_name}.{param_name}",
                        algorithm=request.adapter.adapter_type,
                        component=component_label,
                        component_key=component_key,
                        target_path=lora.lora_name,
                    )
                )

        return refs

    attach_trainable_parameter_provider(adapter, describe_trainable_parameter_refs)
    return adapter


def create_adapter(request: AdapterBuildRequest):
    """Repo-owned wrapper for the built-in DyLoRA adapter runtime."""

    settings = dict(request.adapter.settings)
    adapter_rank = settings.pop("adapter_rank", None)
    adapter_alpha = settings.pop("adapter_alpha", None)
    settings.pop("neuron_dropout", None)

    adapter = legacy_dylora.create_adapter(
        request.context.multiplier,
        adapter_rank,
        adapter_alpha,
        request.context.model.vae,
        request.context.model.text_encoder,
        request.context.model.denoiser,
        **settings,
    )
    adapter.adapter_resolved_targets = request.resolved_targets
    return _attach_trainable_ref_provider(adapter, request)


def create_adapter_from_weights(request: AdapterBuildRequest, weights_path: str):
    """Repo-owned wrapper for loading the built-in DyLoRA adapter from weights."""

    settings = dict(request.adapter.settings)
    settings.pop("adapter_rank", None)
    settings.pop("adapter_alpha", None)
    settings.pop("neuron_dropout", None)

    adapter, weights_sd = legacy_dylora.create_adapter_from_weights(
        request.context.multiplier,
        weights_path,
        request.context.model.vae,
        request.context.model.text_encoder,
        request.context.model.denoiser,
        for_inference=request.context.for_inference,
        **settings,
    )
    adapter.adapter_resolved_targets = request.resolved_targets
    return _attach_trainable_ref_provider(adapter, request), weights_sd
