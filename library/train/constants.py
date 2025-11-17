import re

from torchvision import transforms


EPSILON = 1e-6

HIGH_VRAM = False

TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = "_te_outputs.npz"

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

try:
    import pillow_avif

    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except:
    pass

# JPEG-XL on Linux
try:
    from jxlpy import JXLImagePlugin
    from library.utils.jpeg_xl_util import get_jxl_size

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

# JPEG-XL on Linux and Windows
try:
    import pillow_jxl
    from library.utils.jpeg_xl_util import get_jxl_size

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

# checkpointファイル名
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

DEFAULT_STEP_NAME = "at"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"

# this metadata is referred from train_network and various scripts, so we wrote here
SS_METADATA_KEY_V2 = "ss_v2"
SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
]

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

# Compile the regular expression patterns for float and integer
float_pattern = re.compile(r'''^[+-]?(
    ( (\d+\.\d*) | (\.\d+) ) ([eE][+-]?\d+)?   # Decimal numbers with optional exponent
    | \d+[eE][+-]?\d+                          # Integers with exponent
)$''', re.VERBOSE)

int_pattern = re.compile(r'^[+-]?\d+$')

TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

# DEFAULT_NOISE_OFFSET = 0.0357
