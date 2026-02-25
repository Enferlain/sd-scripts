import json
import argparse
import logging

from safetensors import safe_open


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

with safe_open(args.model, framework="pt") as f:
    metadata = f.metadata()

if metadata is None:
    logger.error("No metadata found")
else:
    # metadata is json dict, but not pretty printed
    # sort by key and pretty print
    print(json.dumps(metadata, indent=4, sort_keys=True))
