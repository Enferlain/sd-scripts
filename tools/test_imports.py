"""
This is a dummy file to test fix_imports.py.
It contains chaotic imports that need sorting.
"""

import sys
import os

# This comment should stick to requests
import requests
from typing import List, Optional, Union, Dict, Any, Sequence, Iterable, Callable, Tuple, Set  # Super long line to test wrapping
from __future__ import annotations  # This should be hoisted to the VERY TOP!
import sys  # Duplicate sys check
import json
import json  # Duplicate json (watch the comments!)

# Third party heavy hitters
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, abort, Blueprint
import numpy as np
import pandas
import django

# Local stuff
from . import sibling
from .. import parent
import library.core
from library.utils import (
    Zed,
    Alpha,
    Beta,
)

# Standard library scattered around
import datetime
from collections import namedtuple

if TYPE_CHECKING:
    # These should stay inside the block
    from library.models import User
    import mypy_extensions

# Conditional imports should be preserved
try:
    import tomllib
except ImportError:
    import tomli as tomllib

import library.auth  # Another first party


def main():
    print("among us esports OMEGALUL!")
