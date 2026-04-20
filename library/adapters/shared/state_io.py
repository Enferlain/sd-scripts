from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from library.training.checkpointing import ResumeState, load_train_state_metadata, save_train_state_metadata

if TYPE_CHECKING:
    from accelerate import Accelerator


@runtime_checkable
class AdapterExportIO(Protocol):
    """Export-style adapter persistence used by the adapter training path.

    This covers explicit adapter weight loads and export/checkpoint artifact
    writes. Training checkpoint state handled by ``accelerator.save_state()``
    remains a distinct flow with its own helper below.
    """

    def load_weights(self, file: str) -> Any: ...

    def save_weights(self, file: str, dtype: Any, metadata: dict[str, str] | None) -> None: ...


AdapterStateIO = AdapterExportIO


@dataclass(frozen=True, slots=True)
class AdapterExportLoadRequest:
    """Context for loading adapter export weights into an existing runtime."""

    file: str


@dataclass(frozen=True, slots=True)
class AdapterExportSaveRequest:
    """Context for writing adapter export weights from an existing runtime."""

    file: str
    dtype: Any
    metadata: dict[str, str] | None = None


def load_adapter_export(adapter: AdapterExportIO, request: AdapterExportLoadRequest) -> Any:
    """Load explicit adapter weights into an existing runtime.

    This is the adapter-training export/load flow used by ``PeftMode`` for
    user-supplied adapter weights, not the accelerator checkpoint-resume flow.
    The return value is adapter-method-defined passthrough information from
    ``adapter.load_weights(...)``.
    """

    if not isinstance(adapter, AdapterExportIO):
        raise TypeError("Adapter export load requires an object implementing AdapterExportIO")
    return adapter.load_weights(request.file)


def save_adapter_export(adapter: AdapterExportIO, request: AdapterExportSaveRequest) -> None:
    """Write an explicit adapter export artifact from an existing runtime.

    This is the adapter-training export/save flow used by ``PeftMode`` for
    final adapter artifacts and adapter-format checkpoints, not
    ``accelerator.save_state()`` resume state.
    """

    if not isinstance(adapter, AdapterExportIO):
        raise TypeError("Adapter export save requires an object implementing AdapterExportIO")
    adapter.save_weights(request.file, request.dtype, request.metadata)


def register_adapter_checkpoint_state_hooks(
    accelerator: Accelerator,
    adapter,
    *,
    save_for_deepspeed: bool,
    current_epoch,
    current_step,
) -> ResumeState:
    """Register adapter-only training checkpoint state hooks.

    This flow is distinct from export-style adapter save/load. The accelerator
    owns the actual wrapped state serialization; the adapter path only narrows
    the checkpoint state to the adapter runtime and stores epoch/step metadata.
    ``save_for_deepspeed`` keeps the DeepSpeed all-process save behavior out of
    adapter-method code and in the training-side orchestration contract.
    """

    resume_state = ResumeState()
    unwrapped_adapter = accelerator.unwrap_model(adapter)

    def _belongs_to_adapter(model) -> bool:
        return accelerator.unwrap_model(model) is unwrapped_adapter

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process or save_for_deepspeed:
            remove_indices = [i for i, model in enumerate(models) if not _belongs_to_adapter(model)]
            for i in reversed(remove_indices):
                if len(weights) > i:
                    weights.pop(i)

            save_train_state_metadata(output_dir, current_epoch, current_step)

    def load_model_hook(models, input_dir):
        remove_indices = [i for i, model in enumerate(models) if not _belongs_to_adapter(model)]
        for i in reversed(remove_indices):
            models.pop(i)

        load_train_state_metadata(input_dir, current_epoch, current_step, resume_state)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    return resume_state
