from __future__ import annotations

from . import impl as legacy_lora
from library.adapters.runtime import AdapterBuildRequest, LoadedAdapterRuntime
from library.adapters.shared import AdapterTrainableParameterRef, attach_trainable_parameter_provider


def _resolve_component_labels(resolved_targets) -> dict[str, str]:
    labels: dict[str, str] = {}
    for target in resolved_targets.targets:
        labels[target.component_key] = target.component
    return labels


def _infer_text_encoder_key(lora_name: str, component_labels: dict[str, str]) -> str:
    if lora_name.startswith(legacy_lora.LoRAAdapter.LORA_PREFIX_TEXT_ENCODER1):
        return "text_encoder1"
    if lora_name.startswith(legacy_lora.LoRAAdapter.LORA_PREFIX_TEXT_ENCODER2):
        return "text_encoder2"

    available_text_encoders = sorted(key for key in component_labels if key.startswith("text_encoder"))
    if not available_text_encoders:
        raise ValueError("LoRA trainable refs require at least one resolved text encoder target")
    return available_text_encoders[0]


def _filter_text_encoder_context(request: AdapterBuildRequest):
    """Preserve TE slot numbering while masking encoders outside resolved targets."""
    text_encoders = request.context.model.text_encoder
    if not isinstance(text_encoders, list):
        return text_encoders

    selected_keys = {
        target.component_key
        for target in request.resolved_targets.targets
        if target.component_key.startswith("text_encoder")
    }
    if not selected_keys:
        return []

    filtered_text_encoders = []
    for index, text_encoder in enumerate(text_encoders, start=1):
        component_key = f"text_encoder{index}"
        filtered_text_encoders.append(text_encoder if component_key in selected_keys else None)
    return filtered_text_encoders


def _attach_trainable_ref_provider(adapter, request: AdapterBuildRequest):
    def describe_trainable_parameter_refs() -> list[AdapterTrainableParameterRef]:
        component_labels = _resolve_component_labels(adapter.adapter_resolved_targets)
        refs: list[AdapterTrainableParameterRef] = []

        for lora in list(getattr(adapter, "text_encoder_loras", [])) + list(getattr(adapter, "unet_loras", [])):
            if lora.lora_name.startswith(legacy_lora.LoRAAdapter.LORA_PREFIX_UNET):
                component_key = "denoiser"
            else:
                component_key = _infer_text_encoder_key(lora.lora_name, component_labels)

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
                        # The legacy built-in wrapper does not retain enough
                        # module-level identity to recover exact shared target
                        # provenance here during the migration.
                        source_target_ref=None,
                    )
                )

        return refs

    attach_trainable_parameter_provider(adapter, describe_trainable_parameter_refs)
    return adapter


def create_adapter(request: AdapterBuildRequest):
    """Repo-owned wrapper for the built-in LoRA adapter runtime."""

    settings = dict(request.adapter.settings)
    adapter_rank = settings.pop("adapter_rank", None)
    adapter_alpha = settings.pop("adapter_alpha", None)
    neuron_dropout = settings.pop("neuron_dropout", None)
    text_encoder = _filter_text_encoder_context(request)

    adapter = legacy_lora.create_adapter(
        request.context.multiplier,
        adapter_rank,
        adapter_alpha,
        request.context.model.vae,
        text_encoder,
        request.context.model.denoiser,
        neuron_dropout=neuron_dropout,
        **settings,
    )
    adapter.adapter_resolved_targets = request.resolved_targets
    return _attach_trainable_ref_provider(adapter, request)


def create_adapter_from_weights(request: AdapterBuildRequest, weights_path: str):
    """Repo-owned wrapper for loading the built-in LoRA adapter from weights."""

    settings = dict(request.adapter.settings)
    settings.pop("adapter_rank", None)
    settings.pop("adapter_alpha", None)
    settings.pop("neuron_dropout", None)
    text_encoder = _filter_text_encoder_context(request)

    adapter, weights_sd = legacy_lora.create_adapter_from_weights(
        request.context.multiplier,
        weights_path,
        request.context.model.vae,
        text_encoder,
        request.context.model.denoiser,
        for_inference=request.context.for_inference,
        **settings,
    )
    adapter.adapter_resolved_targets = request.resolved_targets
    adapter = _attach_trainable_ref_provider(adapter, request)
    return LoadedAdapterRuntime(
        adapter=adapter,
        state=weights_sd,
        _merge_into_impl=lambda merge_request: adapter.merge_to(
            merge_request.model.text_encoder,
            merge_request.model.denoiser,
            weights_sd,
            merge_request.dtype,
            merge_request.device,
        ),
    )
