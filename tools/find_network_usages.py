#!/usr/bin/env python3
"""
Script to find and categorize all "network" occurrences in the codebase.
Outputs categorized results for easy refactoring in PyCharm.
"""

import re
import os
from pathlib import Path
from collections import defaultdict

# Directories to scan (relative to script location)
SCRIPT_DIR = Path(__file__).parent.parent
SCAN_DIRS = [
    SCRIPT_DIR / "library",
    SCRIPT_DIR / "scripts",
    SCRIPT_DIR / "configs",
]
EXCLUDE_DIRS = {"__pycache__", ".git", "venv", "htmlcov", ".mypy_cache"}
EXTENSIONS = {".py", ".yaml", ".yml"}


def find_python_files():
    """Find all Python and YAML files in scan directories."""
    files = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for root, dirs, filenames in os.walk(scan_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for filename in filenames:
                if Path(filename).suffix in EXTENSIONS:
                    files.append(Path(root) / filename)
    return files


def categorize_network_usage(line: str, line_num: int, filepath: Path):
    """Categorize a line containing 'network' into different buckets."""
    results = []
    line_stripped = line.strip()
    
    # Skip comments and docstrings (basic detection)
    if line_stripped.startswith("#"):
        return [("COMMENT", line_stripped, line_num, filepath)]
    
    # Function definitions: def *network*
    func_def = re.search(r'def\s+(\w*network\w*)\s*\(', line, re.IGNORECASE)
    if func_def:
        results.append(("FUNCTION_DEF", func_def.group(1), line_num, filepath))
    
    # Class definitions: class *Network*
    class_def = re.search(r'class\s+(\w*[Nn]etwork\w*)\s*[\(:]', line)
    if class_def:
        results.append(("CLASS_DEF", class_def.group(1), line_num, filepath))
    
    # Function calls: *network*(
    func_calls = re.findall(r'(\w*network\w*)\s*\(', line, re.IGNORECASE)
    for call in func_calls:
        if not re.search(rf'def\s+{call}\s*\(', line):  # Not a definition
            results.append(("FUNCTION_CALL", call, line_num, filepath))
    
    # Variable assignments: network = or network: or self.network
    var_assign = re.findall(r'(\w*network\w*)\s*[=:]', line, re.IGNORECASE)
    for var in var_assign:
        if not re.search(rf'def\s+{var}', line) and not re.search(rf'class\s+{var}', line):
            results.append(("VARIABLE", var, line_num, filepath))
    
    # Parameters: (network, or , network, or network:
    params = re.findall(r'[\(,]\s*(\w*network\w*)\s*[,:=\)]', line, re.IGNORECASE)
    for param in params:
        results.append(("PARAMETER", param, line_num, filepath))
    
    # Dict keys / metadata: "network" or 'network' or SS_*_NETWORK_*
    string_keys = re.findall(r'["\'](\w*network\w*)["\']', line, re.IGNORECASE)
    for key in string_keys:
        results.append(("STRING_KEY", key, line_num, filepath))
    
    # Constants: SS_*_NETWORK_* = 
    constants = re.findall(r'\b(SS_\w*NETWORK\w*)\b', line)
    for const in constants:
        results.append(("CONSTANT", const, line_num, filepath))
    
    # Config access: cfg.network or cfg.peft.network_*
    config_access = re.findall(r'cfg\.(\w*\.)*(\w*network\w*)', line, re.IGNORECASE)
    for match in config_access:
        results.append(("CONFIG_ACCESS", f"cfg.{match[0]}{match[1]}", line_num, filepath))
    
    # Type hints: network: Type or -> Network
    type_hints = re.findall(r':\s*(\w*[Nn]etwork\w*)\b', line)
    for hint in type_hints:
        if hint.lower() != 'network':  # Skip if it's just the variable name
            results.append(("TYPE_HINT", hint, line_num, filepath))
    
    # Imports: from/import *network*
    imports = re.findall(r'(?:from|import)\s+[\w.]*(\w*network\w*)', line, re.IGNORECASE)
    for imp in imports:
        results.append(("IMPORT", imp, line_num, filepath))
    
    # Attribute access: .network or .network_*
    attr_access = re.findall(r'\.(\w*network\w*)\b', line, re.IGNORECASE)
    for attr in attr_access:
        results.append(("ATTRIBUTE", attr, line_num, filepath))
    
    # YAML keys (for .yaml files)
    if filepath.suffix in {".yaml", ".yml"}:
        yaml_keys = re.findall(r'^(\s*\w*network\w*)\s*:', line, re.IGNORECASE)
        for key in yaml_keys:
            results.append(("YAML_KEY", key.strip(), line_num, filepath))
    
    # Catch-all: any remaining 'network' that wasn't categorized
    if not results and re.search(r'\bnetwork', line, re.IGNORECASE):
        results.append(("OTHER", line_stripped[:100], line_num, filepath))
    
    return results


def main():
    files = find_python_files()
    print(f"Scanning {len(files)} files...\n")
    
    all_results = defaultdict(list)
    
    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
        
        for line_num, line in enumerate(lines, 1):
            if "network" in line.lower():
                categories = categorize_network_usage(line, line_num, filepath)
                for category, match, ln, fp in categories:
                    rel_path = fp.relative_to(SCRIPT_DIR)
                    all_results[category].append((match, ln, rel_path))
    
    # Print results by category
    print("=" * 80)
    print("NETWORK USAGE REPORT")
    print("=" * 80)
    
    # Order categories for readability
    category_order = [
        "CLASS_DEF",
        "FUNCTION_DEF", 
        "CONSTANT",
        "YAML_KEY",
        "FUNCTION_CALL",
        "PARAMETER",
        "VARIABLE",
        "ATTRIBUTE",
        "CONFIG_ACCESS",
        "TYPE_HINT",
        "IMPORT",
        "STRING_KEY",
        "COMMENT",
        "OTHER",
    ]
    
    for category in category_order:
        if category not in all_results:
            continue
        items = all_results[category]
        
        print(f"\n{'=' * 40}")
        print(f"{category} ({len(items)} occurrences)")
        print("=" * 40)
        
        # Group by unique name for cleaner output
        by_name = defaultdict(list)
        for match, ln, fp in items:
            by_name[match].append((ln, fp))
        
        for name, locations in sorted(by_name.items()):
            print(f"\n  {name}:")
            for ln, fp in locations[:5]:  # Limit to first 5 locations
                print(f"    - {fp}:{ln}")
            if len(locations) > 5:
                print(f"    ... and {len(locations) - 5} more")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for category in category_order:
        if category in all_results:
            unique = len(set(m for m, _, _ in all_results[category]))
            total = len(all_results[category])
            print(f"  {category}: {unique} unique, {total} total")


if __name__ == "__main__":
    main()
