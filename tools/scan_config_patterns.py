"""
Scan a Python file for cfg/cfg access patterns and categorize them.
Outputs each unique path with its line numbers and content.
"""

import re
import sys
from collections import defaultdict


def scan_config_patterns(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Pattern to match cfg. or cfg. followed by attribute access
    pattern = re.compile(r"\b(cfg|cfg)\.([a-zA-Z_][a-zA-Z0-9_.]*)")

    results = defaultdict(list)

    for i, line in enumerate(lines, 1):
        for match in pattern.finditer(line):
            prefix = match.group(1)
            path = match.group(2)
            key = f"{prefix}.{path}"
            results[key].append((i, line.strip()))

    # Group by top-level category (first part of path)
    categories = defaultdict(dict)
    for key, occurrences in results.items():
        parts = key.split(".")
        if len(parts) >= 2:
            category = parts[1]  # e.g., 'training', 'model', 'performance'
        else:
            category = "root"
        categories[category][key] = occurrences

    # Print grouped output
    print(f"=" * 80)
    print(f"Config Access Patterns in: {filepath}")
    print(f"=" * 80)

    for category in sorted(categories.keys()):
        print(f"\n### {category.upper()} ###")
        print("-" * 40)
        for key in sorted(categories[category].keys()):
            occurrences = categories[category][key]
            print(f"\n  {key} ({len(occurrences)} occurrences)")
            for line_num, content in occurrences:
                # Truncate long lines
                if len(content) > 100:
                    content = content[:97] + "..."
                print(f"    L{line_num:4d}: {content}")

    print(f"\n" + "=" * 80)
    print(f"Total unique patterns: {len(results)}")
    print(f"=" * 80)


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "scripts/sd_textual_inversion.py"
    scan_config_patterns(filepath)
