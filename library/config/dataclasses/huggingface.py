from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HuggingFaceConfig:
    huggingface_repo_id: Optional[str] = None
    huggingface_repo_type: Optional[str] = None
    huggingface_path_in_repo: Optional[str] = None
    huggingface_token: Optional[str] = None
    huggingface_repo_visibility: Optional[str] = None
    save_state_to_huggingface: bool = False
    resume_from_huggingface: bool = False
    async_upload: bool = False
