# fix_imports.py v1.9
# Sorts imports in the order:
# 1. __future__ imports (must stay at top per Python spec)
# 2. Standard library imports (import X)
# 3. Standard library from-imports (from X import Y)
# 4. Third-party imports (import X)
# 5. Third-party from-imports (from X import Y)
# 6. First-party imports (import library.X)
# 7. First-party from-imports (from library.X import Y)
# Then preserves: if TYPE_CHECKING blocks, try/except imports, etc.
#
# Comment heuristics (best-effort):
# - Inline comments on same line stay with that import
# - Comment block immediately preceding an import (no blank line) attaches to it
# - Blank lines separate groups

import importlib.util
import os
import re
import shutil
import sys
import tempfile


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

    is_first_party = "library" in line and (
        line.startswith("import library") or line.startswith("from library")
    )
    is_from = line.startswith("from ")

    if is_first_party:
        return "first_granular" if is_from else "first_module"
    elif is_stdlib(module_path):
        return "std_granular" if is_from else "std_module"
    else:
        return "third_granular" if is_from else "third_module"


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
        "std_module", "std_granular",
        "third_module", "third_granular",
        "first_module", "first_granular",
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
                import_str = "".join(buffer)
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
                    if "(" in line and ")" in line:
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

        subgroup["simple"].sort(key=get_sort_key)
        for comments, imp in subgroup["simple"]:
            if comments:
                output.append(comments)
            output.append(imp)

        subgroup["multiline"].sort(key=get_sort_key)
        for comments, m_imp in subgroup["multiline"]:
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

    # Clean up trailing newlines
    while output and output[-1] == "\n":
        output.pop()
    if output and not output[-1].endswith("\n"):
        output.append("\n")
    
    # One blank line before body (body_lines often starts with blanks already)
    if body_lines:
        output.append("\n")

    # 7. Body
    output.extend(body_lines)

    new_content = "".join(output)

    # Atomic write with backup
    backup_path = file_path + ".bak"
    shutil.copy2(file_path, backup_path)
    
    try:
        dir_name = os.path.dirname(file_path) or "."
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", 
                                          dir=dir_name, delete=False, suffix=".tmp") as tmp:
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
