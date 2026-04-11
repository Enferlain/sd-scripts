import torch


def resolve_state_storage_dtype(state_storage_dtype: str | torch.dtype) -> torch.dtype:
    """Normalize string/dtype state-storage settings onto a concrete torch dtype."""
    if not isinstance(state_storage_dtype, str):
        return state_storage_dtype

    normalized_dtype = state_storage_dtype.strip().lower()
    if normalized_dtype == "float32":
        return torch.float32
    if normalized_dtype == "float16":
        return torch.float16
    if normalized_dtype == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16
