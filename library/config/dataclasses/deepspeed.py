# library/config/dataclasses/deepspeed.py
from dataclasses import dataclass, field

@dataclass
class DeepSpeedConfig:
    deepspeed: bool = field(default=False, metadata={"help": "enable deepspeed training"})
    zero_stage: int = field(default=2, metadata={"help": "Possible options are 0,1,2,3."})
    offload_optimizer_device: str = field(default=None, metadata={"help": "Possible options are none|cpu|nvme."})
    offload_optimizer_nvme_path: str = field(default=None, metadata={"help": "Possible options are /nvme|/local_nvme."})
    offload_param_device: str = field(default=None, metadata={"help": "Possible options are none|cpu|nvme."})
    offload_param_nvme_path: str = field(default=None, metadata={"help": "Possible options are /nvme|/local_nvme."})
    zero3_init_flag: bool = field(default=False, metadata={"help": "Flag to indicate whether to enable `deepspeed.zero.Init` for constructing massive models."})
    zero3_save_16bit_model: bool = field(default=False, metadata={"help": "Flag to indicate whether to save 16-bit model."})
    fp16_master_weights_and_gradients: bool = field(default=False, metadata={"help": "fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32."})
