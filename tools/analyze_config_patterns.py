"""
Config Pattern Analyzer for sd_textual_inversion.py

This script analyzes config usage patterns and generates a report
showing what needs to be updated for consistency with the cfg.* pattern.

Key insight:
- In train() method: can use cfg.* directly since cfg is the parameter
- In helper methods: must use parameter names (training_config, etc.) since cfg isn't passed
"""

import re
from collections import defaultdict


def analyze_file(filepath: str) -> dict:
    """Analyze a Python file for config usage patterns."""

    with open(filepath, encoding="utf-8") as f:
        content = f.read()
        lines = content.split("\n")

    results = {
        "variable_definitions": [],  # Lines like "data_config = cfg.data"
        "method_signatures": [],  # Method definitions with *_config params
        "usages_in_train": [],  # *_config usages inside train() that could be cfg.*
        "usages_in_helpers": [],  # *_config usages in helper methods (must keep)
        "cfg_direct_usages": [],  # Already using cfg.* pattern
        "issues": [],  # Inconsistencies to fix
    }

    # Patterns to look for
    config_vars = [
        "ti_config",
        "training_config",
        "data_config",
        "model_config",
        "optimizer_config",
        "saving_config",
        "performance_config",
        "loss_config",
    ]

    # Track which method we're in
    current_method = None
    in_train_method = False
    indent_level = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip comments and empty lines
        if stripped.startswith("#") or not stripped:
            continue

        # Detect method definitions
        method_match = re.match(r"^(\s*)def (\w+)\s*\(", line)
        if method_match:
            indent_level = len(method_match.group(1))
            current_method = method_match.group(2)
            in_train_method = current_method == "train"

            # Check for *_config in method signature
            for var in config_vars:
                if var in line:
                    results["method_signatures"].append(
                        {
                            "line": i,
                            "method": current_method,
                            "param": var,
                            "content": stripped[:100],
                        }
                    )

        # Detect variable definitions like "data_config = cfg.data"
        for var in config_vars:
            def_match = re.match(rf"^\s*{var}\s*=\s*cfg\.", line)
            if def_match:
                results["variable_definitions"].append(
                    {"line": i, "variable": var, "content": stripped}
                )

        # Detect usages of *_config variables
        for var in config_vars:
            if f"{var}." in line or f"{var}," in line or f"{var})" in line:
                if f"{var} = " not in line:  # Not a definition
                    usage = {
                        "line": i,
                        "variable": var,
                        "method": current_method,
                        "in_train": in_train_method,
                        "content": stripped[:100],
                    }
                    if in_train_method:
                        results["usages_in_train"].append(usage)
                    else:
                        results["usages_in_helpers"].append(usage)

        # Detect direct cfg.* usage
        if "cfg." in line and not any(f"{v} = cfg." in line for v in config_vars):
            results["cfg_direct_usages"].append(
                {"line": i, "method": current_method, "content": stripped[:100]}
            )

    return results


def generate_report(results: dict, output_path: str):
    """Generate a detailed report file."""

    var_to_cfg = {
        "ti_config": "cfg.textual_inversion",
        "training_config": "cfg.training",
        "data_config": "cfg.data",
        "model_config": "cfg.model",
        "optimizer_config": "cfg.optimizer",
        "saving_config": "cfg.output.saving",
        "performance_config": "cfg.performance",
        "loss_config": "cfg.loss",
    }

    report = []
    report.append("=" * 80)
    report.append("CONFIG PATTERN ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Section 1: Variable definitions (can be deleted after refactoring train())
    report.append("=" * 80)
    report.append("1. VARIABLE DEFINITIONS IN train() - TO DELETE AFTER REFACTORING")
    report.append(
        "   These are aliases like 'data_config = cfg.data' that can be removed"
    )
    report.append("=" * 80)
    for item in results["variable_definitions"]:
        report.append(f"  Line {item['line']}: {item['content']}")
    report.append("")

    # Section 2: Method signatures with *_config params
    report.append("=" * 80)
    report.append("2. METHOD SIGNATURES WITH *_config PARAMETERS")
    report.append(
        "   These CANNOT use cfg.* as parameter names (invalid Python syntax)"
    )
    report.append(
        "   Keep as-is, OR refactor method to accept cfg and access internally"
    )
    report.append("=" * 80)
    for item in results["method_signatures"]:
        report.append(
            f"  Line {item['line']} - {item['method']}(): param '{item['param']}'"
        )
        report.append(f"    {item['content']}")
    report.append("")

    # Section 3: Usages in train() - could be refactored
    report.append("=" * 80)
    report.append("3. *_config USAGES IN train() - COULD CHANGE TO cfg.*")
    report.append(
        "   Since train() has 'cfg' parameter, these could become cfg.* directly"
    )
    report.append("=" * 80)

    by_var = defaultdict(list)
    for item in results["usages_in_train"]:
        by_var[item["variable"]].append(item)

    for var, items in sorted(by_var.items()):
        cfg_path = var_to_cfg.get(var, f"cfg.{var}")
        report.append(f"\n  {var} -> {cfg_path}")
        report.append("  " + "-" * 50)
        for item in items:
            report.append(f"    Line {item['line']}: {item['content']}")
    report.append("")

    # Section 4: Usages in helper methods - must keep
    report.append("=" * 80)
    report.append("4. *_config USAGES IN HELPER METHODS - MUST KEEP AS-IS")
    report.append(
        "   These methods receive configs as parameters, so they MUST use those names"
    )
    report.append("=" * 80)

    by_method = defaultdict(list)
    for item in results["usages_in_helpers"]:
        by_method[item["method"]].append(item)

    for method, items in sorted(by_method.items()):
        report.append(f"\n  Method: {method}()")
        report.append("  " + "-" * 50)
        for item in items:
            report.append(
                f"    Line {item['line']} ({item['variable']}): {item['content']}"
            )
    report.append("")

    # Section 5: Already using cfg.* - good!
    report.append("=" * 80)
    report.append("5. ALREADY USING cfg.* DIRECTLY - GOOD!")
    report.append("=" * 80)
    count = len(results["cfg_direct_usages"])
    report.append(f"  Found {count} direct cfg.* usages (not shown for brevity)")
    report.append("")

    # Section 6: Summary and recommendations
    report.append("=" * 80)
    report.append("6. SUMMARY & RECOMMENDATIONS")
    report.append("=" * 80)
    report.append("")
    report.append("  OPTION A: Keep helper method signatures as-is")
    report.append("    - Helper methods keep *_config parameters")
    report.append("    - train() uses *_config aliases (current state)")
    report.append("    - Consistent within the file, but different from other scripts")
    report.append("")
    report.append("  OPTION B: Refactor helper methods to accept cfg")
    report.append(
        "    - Change method params to accept 'cfg' instead of individual configs"
    )
    report.append("    - Inside methods, use cfg.training.*, cfg.output.saving.*, etc.")
    report.append("    - train() can delete the alias variables")
    report.append("    - More consistent with sd_finetune.py pattern")
    report.append("")
    report.append("  RECOMMENDED: Option A is safer, Option B is cleaner but more work")
    report.append("")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    return "\n".join(report)


if __name__ == "__main__":
    import sys

    # Default to sd_textual_inversion.py if no arg given
    filepath = sys.argv[1] if len(sys.argv) > 1 else "scripts/sd_textual_inversion.py"
    output_path = "config_pattern_report.txt"

    print(f"Analyzing: {filepath}")
    results = analyze_file(filepath)
    report = generate_report(results, output_path)

    print(f"\nReport written to: {output_path}")
    print("\n" + "=" * 60)
    print("QUICK SUMMARY:")
    print("=" * 60)
    print(
        f"  Variable definitions to potentially remove: {len(results['variable_definitions'])}"
    )
    print(
        f"  Method signatures with *_config params: {len(results['method_signatures'])}"
    )
    print(f"  Usages in train() (could be cfg.*): {len(results['usages_in_train'])}")
    print(f"  Usages in helpers (must keep): {len(results['usages_in_helpers'])}")
    print(f"  Already using cfg.* pattern: {len(results['cfg_direct_usages'])}")
