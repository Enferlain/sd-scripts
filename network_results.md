================================================================================
NETWORK USAGE REPORT
================================================================================

========================================
PARAMETER (3 occurrences)
========================================

  network:
    - library\training\peft_common.py:212
    - library\training\peft_common.py:238
    - library\training\peft_common.py:256

========================================
VARIABLE (1 occurrences)
========================================

  network:
    - library\training\peft_common.py:221

========================================
COMMENT (2 occurrences)
========================================

  # TODO support Hypernetworks:
    - library\models\original_unet.py:668
    - library\models\sdxl_original_unet.py:473

========================================
OTHER (2 occurrences)
========================================

  "ss_adapter_module": cfg.peft.module,  # NETWORK REFACTOR:
    - library\training\peft_common.py:610

  logger.info(f"network multiplier: {example['adapter_multipliers'][j]}"):
    - library\data\dataset_utils.py:177

================================================================================
SUMMARY
================================================================================
  PARAMETER: 1 unique, 3 total
  VARIABLE: 1 unique, 1 total
  COMMENT: 1 unique, 2 total
  OTHER: 2 unique, 2 total