"""
This is a dummy file to test fix_imports.py.
It contains chaotic imports that need sorting.
"""


# This comment should stick to requests
from __future__ import annotations  # This should be hoisted to the VERY TOP!

# Third party heavy hitters

# Local stuff

# Standard library scattered around

if TYPE_CHECKING:
    # These should stay inside the block
    pass

# Conditional imports should be preserved
try:
    import tomllib
except ImportError:
    pass



def main():
    print("among us esports OMEGALUL!")
