from typing import Literal


UPDATE_STRATEGY = Literal["unmodified", "cautious", "grams", "both"]
CLIP_TYPE = Literal["unit", "layer", "element"]
STATE_PRECISION = Literal["parameter", "q4bit", "q8bit", "qfp8"]
