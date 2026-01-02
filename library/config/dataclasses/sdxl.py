from dataclasses import dataclass

@dataclass
class SDXLConfig:
    """
    SDXL-specific configuration.
    
    Note: Most fields have been migrated to other configs:
    - cache_text_encoder_outputs -> PerformanceConfig
    - cache_text_encoder_outputs_to_disk -> PerformanceConfig
    - disable_mmap_load_safetensors -> PerformanceConfig
    - train_text_encoder -> REMOVED (use LR-based control via optimizer.learning_rates.text_encoders)
    - fused_optimizer_groups -> OptimizerConfig
    
    This class is kept for future SDXL-specific settings that don't belong elsewhere.
    """
    pass
