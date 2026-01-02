# fix_imports.py v2.2
#
# Sorts and organizes Python imports according to PEP8 style.
#
# Import Order:
#   1. __future__ imports (must stay at top per Python spec)
#   2. Standard library imports (import X)
#   3. Standard library from-imports (from X import Y)
#   4. Third-party imports (import X)
#   5. Third-party from-imports (from X import Y)
#   6. First-party imports (import library.X)
#   7. First-party from-imports (from library.X import Y)
#   8. Relative imports (from . import X, from .. import Y)
#   9. TYPE_CHECKING blocks (preserved as-is)
#  10. Conditional imports (try/except blocks, preserved as-is)
#
# Features:
#   - Deduplication: Removes duplicate imports at top-level and inside multiline
#     brackets. Handles imports with inline comments (e.g., `import x  # note`)
#   - Combining: Merges simple `import X` statements into `import a, b, c` format
#     while respecting the max line length (88 chars by default)
#   - Auto-wrapping: Converts long `from X import a, b, c` into multiline format
#     with parentheses when they exceed the line length limit
#   - Sorting: Sorts module names case-insensitively, with original case as
#     tie-breaker for stability. Sorts items inside multiline import brackets.
#   - Comment preservation: Keeps inline comments attached to their imports,
#     and standalone comments are associated with the following import
#   - Alias preservation: Imports with `as` aliases are kept separate (not combined)
#
# Usage:
#   python fix_imports.py <file_path>

import importlib.util
import os
import re
import shutil
import sys
import tempfile

# Configuration
MAX_LINE_LENGTH = 88  # Black-compatible default


def dedupe_simple_imports(imports: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove duplicate imports, merging comments from duplicates onto the first occurrence."""
    seen: dict[str, int] = {}  # key -> index in result
    result: list[tuple[str, str]] = []

    for comments, imp in imports:
        # Extract the module name for dedup key
        first_line = imp.split("\n")[0] if "\n" in imp else imp
        match = re.match(r"^(?:from|import)\s+([^\s\(,]+)", first_line.strip())

        if match:
            key = match.group(1).lower()
            if key not in seen:
                seen[key] = len(result)
                result.append((comments, imp))
            elif comments.strip():
                # Duplicate with comments - merge comments onto first occurrence
                idx = seen[key]
                existing_comments, existing_imp = result[idx]
                merged_comments = existing_comments + comments
                result[idx] = (merged_comments, existing_imp)
            # else: duplicate without comments - just skip
        else:
            result.append((comments, imp))

    return result


def combine_simple_imports(imports: list[tuple[str, str]], max_len: int = MAX_LINE_LENGTH) -> list[tuple[str, str]]:
    """Combine simple 'import X' statements into 'import a, b, c' when possible.

    Only combines imports without comments. Respects line length limit.
    """
    # Separate imports with and without comments
    with_comments = [(c, i) for c, i in imports if c.strip()]
    without_comments = [(c, i) for c, i in imports if not c.strip()]

    # Extract module names from comment-less imports
    modules = []
    for _, imp in without_comments:
        stripped = imp.strip().rstrip("\n")
        # Only handle simple 'import X' (not 'from X import Y')
        if stripped.startswith("import ") and " as " not in stripped and "#" not in stripped:
            # Could be 'import a, b, c' already
            parts = stripped[7:].split(",")
            for part in parts:
                mod = part.strip()
                if mod:
                    modules.append(mod)
        else:
            # Keep as-is (has alias, comment, or is a 'from' import)
            with_comments.append(("", imp))

    if not modules:
        return with_comments

    # Sort and dedupe modules
    modules_set = {m.lower() for m in modules}
    modules = sorted(set(modules), key=str.lower)

    # Filter out with_comments entries whose modules are already in the combined set
    # This handles cases like: import json  # comment (when json is already combined)
    filtered_with_comments = []
    for c, imp in with_comments:
        # Extract the module name from imports like "import X  # comment" or "import X as Y"
        stripped = imp.strip().rstrip("\n")
        if stripped.startswith("import "):
            # Get the module name (first word after "import")
            import_part = stripped[7:].split("#")[0].strip()  # Remove inline comment
            mod_name = import_part.split(" as ")[0].strip()  # Handle aliases
            mod_name = mod_name.split(",")[0].strip()  # Handle multi-imports
            if mod_name.lower() in modules_set:
                # This module is already included in combined imports - skip it
                # But if it has preceding comments, we should attach them to the combined line
                continue
        filtered_with_comments.append((c, imp))

    # Combine into lines respecting max length
    combined = []
    current_line_modules = []
    current_len = len("import ")

    for mod in modules:
        add_len = len(mod) + (2 if current_line_modules else 0)  # ", " separator
        if current_len + add_len > max_len and current_line_modules:
            # Flush current line
            combined.append(("", "import " + ", ".join(current_line_modules) + "\n"))
            current_line_modules = [mod]
            current_len = len("import ") + len(mod)
        else:
            current_line_modules.append(mod)
            current_len += add_len

    if current_line_modules:
        combined.append(("", "import " + ", ".join(current_line_modules) + "\n"))

    return filtered_with_comments + combined


def wrap_long_from_import(comments: str, imp: str, max_len: int = MAX_LINE_LENGTH) -> tuple[str, str]:
    """Wrap a long 'from X import a, b, c' into multiline format if needed."""
    first_line = imp.split("\n")[0] if "\n" in imp else imp
    stripped = first_line.strip()

    # Skip if already multiline or not a from-import
    if "(" in stripped or not stripped.startswith("from ") or " import " not in stripped:
        return comments, imp

    # Skip if within length limit
    if len(stripped) <= max_len:
        return comments, imp

    # Parse the import
    match = re.match(r"^from\s+(\S+)\s+import\s+(.+)$", stripped)
    if not match:
        return comments, imp

    module, items_str = match.groups()

    # Handle inline comment
    inline_comment = ""
    if "#" in items_str:
        items_str, inline_comment = items_str.split("#", 1)
        inline_comment = "  #" + inline_comment

    items = [item.strip() for item in items_str.split(",") if item.strip()]

    # Build multiline format
    lines = [f"from {module} import (\n"]
    for item in sorted(items, key=str.lower):
        lines.append(f"    {item},\n")
    lines.append(f"){inline_comment}\n" if inline_comment else ")\n")

    return comments, "".join(lines)


def is_stdlib(module_name: str) -> bool:
    """Check if a module is part of the standard library."""
    if not module_name or module_name.startswith("."):
        return False

    base_module = module_name.split(".")[0]
    if not base_module:
        return False

    if base_module in sys.builtin_module_names:
        return True

    try:
        spec = importlib.util.find_spec(base_module)
    except (ModuleNotFoundError, ValueError):
        return False

    if spec is None:
        return False

    origin = spec.origin or ""
    return "site-packages" not in origin and "dist-packages" not in origin


def classify_import(line: str) -> str | None:
    """Classify an import line into a category."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("from __future__"):
        return "future"

    if line.startswith("from .") or (line.startswith("from ") and " import " in line):
        match = re.match(r"^from\s+([^\s]+)", line)
        if match and match.group(1).startswith("."):
            return "relative"

    match = re.match(r"^(?:from|import)\s+([^\s\(]+)", line)
    module_path = match.group(1) if match else ""

    is_first_party = "library" in line and (line.startswith("import library") or line.startswith("from library"))
    is_from = line.startswith("from ")

    if is_first_party:
        return "first_granular" if is_from else "first_module"
    elif is_stdlib(module_path):
        return "std_granular" if is_from else "std_module"
    else:
        return "third_granular" if is_from else "third_module"


def sort_multiline_import(import_str: str) -> str:
    """Sort the imports inside a multiline import statement.

    Handles:
        from module import (
            Zed,
            Alpha,
            Beta,
        )

    Becomes:
        from module import (
            Alpha,
            Beta,
            Zed,
        )
    """
    # Check if this is a multiline import with parentheses
    if "(" not in import_str or ")" not in import_str:
        return import_str

    lines = import_str.splitlines(keepends=True)

    # Find the opening line with "("
    open_idx = -1
    for i, line in enumerate(lines):
        if "(" in line:
            open_idx = i
            break

    if open_idx == -1:
        return import_str

    # Find the closing line with ")"
    close_idx = -1
    for i, line in enumerate(lines):
        if ")" in line:
            close_idx = i
            break

    if close_idx == -1 or close_idx <= open_idx:
        return import_str

    # Check if it's a single-line import like: from x import (a, b, c)
    if open_idx == close_idx:
        # Single line with parentheses - sort inline
        line = lines[open_idx]
        match = re.match(r"^(.*?\()(.*)(\).*)$", line)
        if match:
            prefix, items_str, suffix = match.groups()
            items = [item.strip() for item in items_str.split(",") if item.strip()]
            items.sort(key=str.lower)
            sorted_items = ", ".join(items)
            lines[open_idx] = f"{prefix}{sorted_items}{suffix}"
        return "\n".join(lines)

    # Collect import items (lines between open and close)
    # Each item is (sort_key_lower, sort_key_original, normalized_line)
    import_items = []
    pending_comment_lines = []  # Standalone comments attach to the next import
    seen_items = set()  # For duplicate detection

    for i in range(open_idx + 1, close_idx):
        line = lines[i]
        item_line = line.strip()

        if not item_line:
            # Blank line - keep pending comments, skip the blank
            continue

        if item_line.startswith("#"):
            # Standalone comment line - attach to next import item
            pending_comment_lines.append(line)
            continue

        # This is an import item
        # Extract the item name (without comma) for sorting and deduplication
        item = item_line.rstrip(",").strip()

        # Handle inline comments: extract just the import part for dedup key
        if "#" in item:
            dedup_key = item.split("#", 1)[0].rstrip(",").strip().lower()
        else:
            dedup_key = item.lower()

        # Skip duplicates
        if dedup_key in seen_items:
            pending_comment_lines = []  # Discard comments for duplicate
            continue
        seen_items.add(dedup_key)

        if item:
            # Normalize: ensure trailing comma on item
            # Preserve original indentation
            leading_ws = line[: len(line) - len(line.lstrip())]

            # Check for inline comment
            if "#" in item_line:
                parts = item_line.split("#", 1)
                item_part = parts[0].rstrip(",").strip()
                comment_part = "#" + parts[1]
                normalized_line = f"{leading_ws}{item_part},  {comment_part}\n"
                sort_key = item_part  # Sort by import name, not comment
            else:
                normalized_line = f"{leading_ws}{item.rstrip(',')},\n"
                sort_key = item

            # Prepend any pending standalone comments
            if pending_comment_lines:
                normalized_line = "".join(pending_comment_lines) + normalized_line
                pending_comment_lines = []

            # Use (lowercase, original) for stable case-insensitive sort
            import_items.append((sort_key.lower(), sort_key, normalized_line))

    # Sort by the item name (case-insensitive, with original as tie-breaker)
    import_items.sort(key=lambda x: (x[0], x[1]))

    # Reconstruct
    result_lines = lines[: open_idx + 1]  # Keep lines up to and including "("
    for _, _, normalized_line in import_items:
        result_lines.append(normalized_line)

    # Add any trailing orphan comments (rare but possible)
    result_lines.extend(pending_comment_lines)
    result_lines.extend(lines[close_idx:])  # Keep closing ")" and after

    return "".join(result_lines)


def get_sort_key(import_entry: tuple[str, str]) -> str:
    """Get the module name for sorting. Entry is (comments, import_str)."""
    import_str = import_entry[1]
    first_line = import_str.split("\n")[0] if "\n" in import_str else import_str
    match = re.match(r"^(?:from|import)\s+([^\s\(]+)", first_line.strip())
    return match.group(1).lower() if match else import_str.lower()


def is_top_level(line: str) -> bool:
    """Check if this line has no indentation."""
    return line and not line[0].isspace()


def collect_block(lines: list[str], start_idx: int) -> tuple[list[str], int]:
    """Collect an indented block (like if TYPE_CHECKING: or try:)."""
    block = [lines[start_idx]]
    i = start_idx + 1

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "":
            block.append(line)
            i += 1
            continue

        if line[0].isspace() or stripped.startswith(("except ", "except:", "elif ", "else:")):
            block.append(line)
            i += 1
        else:
            break

    return block, i


def sort_imports_in_file(file_path: str) -> None:
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    categories = [
        "std_module",
        "std_granular",
        "third_module",
        "third_granular",
        "first_module",
        "first_granular",
    ]
    # Each group stores tuples of (preceding_comments, import_line_or_block)
    groups = {k: {"simple": [], "multiline": []} for k in categories}

    header_lines = []
    future_imports = []  # List of (comments, import)
    relative_imports = []  # List of (comments, import)
    type_checking_blocks = []
    conditional_import_blocks = []
    body_lines = []

    pending_comments = []  # Comments waiting to attach to next import
    buffer = []
    inside_multiline = False
    found_first_import = False
    imports_ended = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle multiline imports
        if inside_multiline:
            buffer.append(line)
            if ")" in line:
                inside_multiline = False
                cat = classify_import(buffer[0])
                import_str = sort_multiline_import("".join(buffer))
                comment_str = "".join(pending_comments)
                pending_comments = []

                if cat == "future":
                    future_imports.append((comment_str, import_str))
                elif cat == "relative":
                    relative_imports.append((comment_str, import_str))
                elif cat and cat in groups:
                    groups[cat]["multiline"].append((comment_str, import_str))
                buffer = []
            i += 1
            continue

        # TYPE_CHECKING block
        if stripped.startswith("if TYPE_CHECKING") and is_top_level(line):
            block, i = collect_block(lines, i)
            # Attach pending comments to block
            block_str = "".join(pending_comments) + "".join(block)
            pending_comments = []
            type_checking_blocks.append(block_str)
            found_first_import = True
            continue

        # try: block
        if stripped.startswith("try:") and is_top_level(line):
            block, i = collect_block(lines, i)
            block_str = "".join(pending_comments) + "".join(block)
            pending_comments = []
            conditional_import_blocks.append(block_str)
            found_first_import = True
            continue

        is_import_line = stripped.startswith("import ") or stripped.startswith("from ")

        if is_import_line and is_top_level(line) and not imports_ended:
            found_first_import = True

            if "(" in stripped and ")" not in stripped:
                inside_multiline = True
                buffer.append(line)
            else:
                cat = classify_import(line)
                comment_str = "".join(pending_comments)
                pending_comments = []

                if cat == "future":
                    future_imports.append((comment_str, line))
                elif cat == "relative":
                    relative_imports.append((comment_str, line))
                elif cat:
                    # Check for parentheses in import statement (not in comments)
                    code_part = line.split("#")[0]
                    if "(" in code_part and ")" in code_part:
                        groups[cat]["multiline"].append((comment_str, line))
                    else:
                        groups[cat]["simple"].append((comment_str, line))
                else:
                    body_lines.extend([comment_str, line] if comment_str else [line])
        elif not found_first_import:
            header_lines.append(line)
        else:
            # After imports started
            if stripped == "":
                # Blank line - flush pending comments to body if no more imports
                has_more = False
                for j in range(i + 1, min(i + 10, len(lines))):
                    fs = lines[j].strip()
                    if fs.startswith(("import ", "from ", "if TYPE_CHECKING", "try:")):
                        if is_top_level(lines[j]):
                            has_more = True
                            break
                    elif fs and not fs.startswith("#"):
                        break

                if not has_more:
                    imports_ended = True
                    # Flush pending comments
                    body_lines.extend(pending_comments)
                    pending_comments = []
                    body_lines.append(line)
                else:
                    # Blank line separates comment groups - flush pending
                    body_lines.extend(pending_comments)
                    pending_comments = []

            elif stripped.startswith("#") and not imports_ended:
                # Comment - check if more imports follow immediately
                has_more_immediate = False
                j = i + 1
                while j < len(lines) and lines[j].strip().startswith("#"):
                    j += 1  # Skip consecutive comments
                if j < len(lines):
                    next_stripped = lines[j].strip()
                    if next_stripped.startswith(("import ", "from ", "if TYPE_CHECKING", "try:")):
                        if is_top_level(lines[j]):
                            has_more_immediate = True

                if has_more_immediate:
                    # Attach to next import
                    pending_comments.append(line)
                else:
                    # No import follows - send to body
                    imports_ended = True
                    body_lines.extend(pending_comments)
                    pending_comments = []
                    body_lines.append(line)
            else:
                imports_ended = True
                body_lines.extend(pending_comments)
                pending_comments = []
                body_lines.append(line)

        i += 1

    # Build output
    output = []

    # 1. Header
    output.extend(header_lines)

    # 2. __future__ imports FIRST
    for comments, imp in future_imports:
        if comments:
            output.append(comments)
        output.append(imp)

    # 3. Sorted imports by category
    prev_had_content = len(future_imports) > 0
    for key in categories:
        subgroup = groups[key]
        if not subgroup["simple"] and not subgroup["multiline"]:
            continue

        if prev_had_content:
            output.append("\n")

        # Process simple imports
        simple_imports = subgroup["simple"]
        simple_imports.sort(key=get_sort_key)
        simple_imports = dedupe_simple_imports(simple_imports)

        # For module imports (import X), combine into single lines
        if key.endswith("_module"):
            simple_imports = combine_simple_imports(simple_imports)

        # For granular imports (from X import Y), wrap long lines
        if key.endswith("_granular"):
            simple_imports = [wrap_long_from_import(c, i) for c, i in simple_imports]

        for comments, imp in simple_imports:
            if comments:
                output.append(comments)
            output.append(imp)

        # Process multiline imports (already sorted inside by sort_multiline_import)
        multiline_imports = subgroup["multiline"]
        multiline_imports.sort(key=get_sort_key)
        multiline_imports = dedupe_simple_imports(multiline_imports)

        for comments, m_imp in multiline_imports:
            if output and output[-1] != "\n" and not output[-1].endswith("\n\n"):
                output.append("\n")
            if comments:
                output.append(comments)
            output.append(m_imp)

        prev_had_content = True

    # 4. Relative imports (preserve order)
    if relative_imports:
        if prev_had_content:
            output.append("\n")
        for comments, imp in relative_imports:
            if comments:
                output.append(comments)
            output.append(imp)
        prev_had_content = True

    # 5. TYPE_CHECKING blocks
    for block in type_checking_blocks:
        if output and not output[-1].endswith("\n\n"):
            if not output[-1].endswith("\n"):
                output.append("\n")
            output.append("\n")
        output.append(block)

    # 6. Conditional import blocks
    for block in conditional_import_blocks:
        if output and not output[-1].endswith("\n\n"):
            if not output[-1].endswith("\n"):
                output.append("\n")
            output.append("\n")
        output.append(block)

    # 7. Body - ensure exactly 2 blank lines before body (PEP8)
    if body_lines:
        # Strip leading blank lines from body
        while body_lines and body_lines[0].strip() == "":
            body_lines.pop(0)

        # Only add spacing if we actually have imports above!
        if output:
            # Strip all trailing newlines from output (clean up previous loops)
            while output and output[-1] == "\n":
                output.pop()

            # Ensure the last import line ends with exactly one newline
            if output and output[-1].endswith("\n"):
                output[-1] = output[-1].rstrip("\n") + "\n"

            # Add 2 newlines (End of line + 2 blanks = 3 \n total)
            output.append("\n\n")

        output.extend(body_lines)
    new_content = "".join(output)

    # Atomic write with backup
    backup_path = file_path + ".bak"
    shutil.copy2(file_path, backup_path)

    try:
        dir_name = os.path.dirname(file_path) or "."
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=dir_name, delete=False, suffix=".tmp") as tmp:
            tmp.write(new_content)
            tmp_path = tmp.name

        shutil.move(tmp_path, file_path)
        os.remove(backup_path)
        print(f"Fixed imports in {file_path}")

    except Exception as e:
        if os.path.exists(backup_path):
            shutil.move(backup_path, file_path)
        print(f"Error fixing {file_path}: {e}. Original restored from backup.")
        raise


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        sort_imports_in_file(sys.argv[1])
    else:
        print("Usage: python fix_imports.py <file_path>")
