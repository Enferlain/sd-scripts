"""
Config Pattern Audit Script

Scans all Python files in configs/, library/, scripts/, tests/ and reports:
- File type (script, strategy, library utility, test)
- Current pattern usage (*_config vs cfg.*)
- Expected pattern based on file type
- Discrepancies to fix

Usage: python tools/audit_config_patterns.py
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Folders to scan
SCAN_FOLDERS = ["configs", "library", "scripts", "tests"]

# Skip patterns
SKIP_PATTERNS = ["venv", "__pycache__", ".git", "node_modules"]

# Config variable patterns
CONFIG_VAR_PATTERN = re.compile(
    r"\b(ti_config|training_config|data_config|model_config|optimizer_config|saving_config|performance_config|loss_config|peft_config|output_config|scheduler_config)\b"
)
CFG_DOT_PATTERN = re.compile(
    r"\bcfg\.(training|model|optimizer|performance|data|output|loss|peft|validation|timesteps|textual_inversion|sdxl)\b"
)


@dataclass
class FileAnalysis:
    filepath: str
    file_type: str  # 'script', 'strategy', 'library', 'test', 'config'
    expected_pattern: str  # 'cfg.*', 'typed_params', 'either'
    config_var_count: int
    cfg_dot_count: int
    config_var_lines: List[Tuple[int, str]]
    cfg_dot_lines: List[Tuple[int, str]]
    has_discrepancy: bool
    notes: str


def determine_file_type(filepath: str) -> Tuple[str, str, str]:
    """Determine file type and expected pattern based on location and content."""
    path = Path(filepath)
    parts = path.parts

    # Check folder
    if "scripts" in parts:
        return "script", "cfg.*", "Scripts should use cfg.* in train() function"

    if "strategies" in parts:
        return (
            "strategy",
            "cfg.*",
            "Strategies receive cfg and should use cfg.* pattern",
        )

    if "tests" in parts:
        return "test", "either", "Tests can use either pattern as needed"

    if "configs" in parts:
        return "config", "either", "Config YAML/dataclass files"

    if "library" in parts:
        # Check if it's a strategy-like file
        if "strategy" in path.name.lower():
            return "strategy", "cfg.*", "Strategy files use cfg.* pattern"

        # Check for common library utility patterns
        return (
            "library",
            "typed_params",
            "Library utilities should use typed individual params",
        )

    return "unknown", "unknown", ""


def analyze_file(filepath: str) -> FileAnalysis:
    """Analyze a single Python file for config patterns."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.split("\n")
    except Exception as e:
        return FileAnalysis(
            filepath=filepath,
            file_type="error",
            expected_pattern="unknown",
            config_var_count=0,
            cfg_dot_count=0,
            config_var_lines=[],
            cfg_dot_lines=[],
            has_discrepancy=False,
            notes=f"Error reading file: {e}",
        )

    file_type, expected_pattern, notes = determine_file_type(filepath)

    # Find config_* usages
    config_var_lines = []
    cfg_dot_lines = []

    for i, line in enumerate(lines, 1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # Check for config_* pattern
        if CONFIG_VAR_PATTERN.search(line):
            config_var_lines.append((i, stripped[:80]))

        # Check for cfg.* pattern
        if CFG_DOT_PATTERN.search(line):
            cfg_dot_lines.append((i, stripped[:80]))

    config_var_count = len(config_var_lines)
    cfg_dot_count = len(cfg_dot_lines)

    # Determine if there's a discrepancy
    has_discrepancy = False
    if expected_pattern == "cfg.*" and config_var_count > 0:
        # Scripts/strategies using config_* when they should use cfg.*
        has_discrepancy = True
    elif expected_pattern == "typed_params" and cfg_dot_count > 0:
        # Library utilities using cfg.* when they should use typed params
        # This is actually OK if it's in function signatures receiving cfg
        # We'll flag it for review
        has_discrepancy = True

    return FileAnalysis(
        filepath=filepath,
        file_type=file_type,
        expected_pattern=expected_pattern,
        config_var_count=config_var_count,
        cfg_dot_count=cfg_dot_count,
        config_var_lines=config_var_lines,
        cfg_dot_lines=cfg_dot_lines,
        has_discrepancy=has_discrepancy,
        notes=notes,
    )


def scan_folders(base_path: str, folders: List[str]) -> List[FileAnalysis]:
    """Scan all Python files in specified folders."""
    results = []

    for folder in folders:
        folder_path = Path(base_path) / folder
        if not folder_path.exists():
            continue

        for root, dirs, files in os.walk(folder_path):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in SKIP_PATTERNS]

            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    analysis = analyze_file(filepath)
                    results.append(analysis)

    return results


def generate_report(results: List[FileAnalysis], output_path: str):
    """Generate comprehensive audit report."""
    report = []

    report.append("=" * 100)
    report.append("CONFIG PATTERN AUDIT REPORT")
    report.append("=" * 100)
    report.append("")
    report.append("Pattern Guide:")
    report.append(
        "  - Scripts/Strategies: Should use cfg.* (e.g., cfg.training.max_train_steps)"
    )
    report.append(
        "  - Library utilities: Should use typed params (e.g., training_config: TrainingConfig)"
    )
    report.append("")

    # Group by file type
    by_type = defaultdict(list)
    for r in results:
        by_type[r.file_type].append(r)

    # Summary
    report.append("=" * 100)
    report.append("SUMMARY BY FILE TYPE")
    report.append("=" * 100)

    for file_type in ["script", "strategy", "library", "test", "config"]:
        files = by_type.get(file_type, [])
        if not files:
            continue

        discrepancies = sum(1 for f in files if f.has_discrepancy)
        total_config_var = sum(f.config_var_count for f in files)
        total_cfg_dot = sum(f.cfg_dot_count for f in files)

        report.append(f"\n{file_type.upper()} ({len(files)} files)")
        report.append(f"  config_* usages: {total_config_var}")
        report.append(f"  cfg.* usages: {total_cfg_dot}")
        report.append(f"  Files with potential issues: {discrepancies}")

    # Files with discrepancies
    report.append("")
    report.append("=" * 100)
    report.append("FILES WITH POTENTIAL ISSUES (need review)")
    report.append("=" * 100)

    for file_type in ["script", "strategy", "library"]:
        files = [f for f in by_type.get(file_type, []) if f.has_discrepancy]
        if not files:
            continue

        report.append(f"\n--- {file_type.upper()} ---")

        for f in files:
            rel_path = f.filepath.replace("\\", "/")
            report.append(f"\n  File: {rel_path}")
            report.append(f"  Expected: {f.expected_pattern}")
            report.append(
                f"  config_* count: {f.config_var_count}, cfg.* count: {f.cfg_dot_count}"
            )

            if f.config_var_count > 0 and f.expected_pattern == "cfg.*":
                report.append(f"  ISSUE: Using config_* when should use cfg.*")
                report.append(f"  Sample lines with config_*:")
                for line_num, content in f.config_var_lines[:5]:
                    report.append(f"    L{line_num}: {content}")
                if len(f.config_var_lines) > 5:
                    report.append(f"    ... and {len(f.config_var_lines) - 5} more")

    # Detailed breakdown for scripts
    report.append("")
    report.append("=" * 100)
    report.append("DETAILED SCRIPT ANALYSIS")
    report.append("=" * 100)

    for f in sorted(by_type.get("script", []), key=lambda x: x.filepath):
        if f.config_var_count == 0 and f.cfg_dot_count == 0:
            continue

        rel_path = f.filepath.replace("\\", "/")
        status = "✅" if not f.has_discrepancy else "⚠️"
        report.append(f"\n{status} {rel_path}")
        report.append(f"   config_*: {f.config_var_count} | cfg.*: {f.cfg_dot_count}")

    # Detailed breakdown for library
    report.append("")
    report.append("=" * 100)
    report.append("DETAILED LIBRARY ANALYSIS")
    report.append("=" * 100)

    for f in sorted(by_type.get("library", []), key=lambda x: x.filepath):
        if f.config_var_count == 0 and f.cfg_dot_count == 0:
            continue

        rel_path = f.filepath.replace("\\", "/")
        # For library files, config_* is expected, cfg.* might be OK if receiving cfg param
        if f.cfg_dot_count > 0:
            status = "🔍"  # Needs review
        else:
            status = "✅"
        report.append(f"\n{status} {rel_path}")
        report.append(f"   config_*: {f.config_var_count} | cfg.*: {f.cfg_dot_count}")

    # Write report
    report_text = "\n".join(report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


if __name__ == "__main__":
    import sys

    base_path = "."  # Run from project root
    output_path = "config_audit_report.txt"

    print("Scanning folders:", SCAN_FOLDERS)
    results = scan_folders(base_path, SCAN_FOLDERS)

    print(f"Analyzed {len(results)} Python files")
    report = generate_report(results, output_path)

    print(f"\nReport written to: {output_path}")

    # Quick console summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)

    discrepancies = [r for r in results if r.has_discrepancy]
    print(f"Total files analyzed: {len(results)}")
    print(f"Files with potential issues: {len(discrepancies)}")

    if discrepancies:
        print("\nFiles to review:")
        for f in discrepancies[:10]:
            rel_path = f.filepath.replace("\\", "/")
            print(f"  - {rel_path} ({f.file_type})")
        if len(discrepancies) > 10:
            print(f"  ... and {len(discrepancies) - 10} more")
