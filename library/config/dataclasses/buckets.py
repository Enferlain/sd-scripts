from dataclasses import dataclass, field

@dataclass
class BucketsConfig:
    enable_bucket: bool = field(default=False, metadata={"help": "enable buckets for multi aspect ratio training"})
    min_bucket_reso: int = field(default=256, metadata={"help": "minimum resolution for buckets"})
    max_bucket_reso: int = field(default=1024, metadata={"help": "maximum resolution for buckets"})
    bucket_reso_steps: int = field(default=64, metadata={"help": "steps of resolution for buckets"})
    bucket_no_upscale: bool = field(default=False, metadata={"help": "make bucket for each image without upscaling"})
