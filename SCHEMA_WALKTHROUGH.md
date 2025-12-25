# PyCharm Refactoring Guide for Config Schema Migration

## What PyCharm Can Do Well

### 1. Rename Dataclass Files (Move/Rename)

**Example**: Rename [network.py](file:///d:/Projects/sd-scripts/library/config/dataclasses/network.py) → [peft.py](file:///d:/Projects/sd-scripts/scripts/sd_peft.py)

1. Right-click [library/config/dataclasses/network.py](file:///d:/Projects/sd-scripts/library/config/dataclasses/network.py) in Project view
2. **Refactor → Rename** (Shift+F6)
3. Enter new name: [peft.py](file:///d:/Projects/sd-scripts/scripts/sd_peft.py)
4. PyCharm will update all `from ... import` statements

> ⚠️ This won't update YAML files or string references like `"network"`

---

### 2. Rename Classes

**Example**: Rename [NetworkConfig](file:///d:/Projects/sd-scripts/library/config/dataclasses/network.py#4-26) → [PeftConfig](file:///d:/Projects/sd-scripts/library/config/dataclasses/sd_peft.py#20-38)

1. Place cursor on `class NetworkConfig:`
2. **Refactor → Rename** (Shift+F6)
3. Enter: [PeftConfig](file:///d:/Projects/sd-scripts/library/config/dataclasses/sd_peft.py#20-38)
4. Check "Search in comments and strings" for extra coverage
5. Preview changes, then apply

---

### 3. Rename Fields (with type hints)

**Example**: Rename `network_dim` → `dim`

1. Place cursor on the field name in the dataclass
2. **Refactor → Rename** (Shift+F6)
3. Enter new name

✅ **Works well if**: Code uses typed config objects (`cfg: SDPeftConfig`)
⚠️ **Misses**: Untyped access, YAML keys, string literals

---

### 4. Move Fields Between Classes

**Example**: Move `unet_lr` from [NetworkConfig](file:///d:/Projects/sd-scripts/library/config/dataclasses/network.py#4-26) → [OptimizerConfig](file:///d:/Projects/sd-scripts/library/config/dataclasses/optimizer.py#4-24)

PyCharm can't auto-move fields between dataclasses. Manual steps:

1. Copy field definition to target class
2. Use **Find Usages** (Alt+F7) on old field to find all references
3. Update each reference manually
4. Delete old field

---

### 5. Find All Usages

Before any rename, check impact:

1. Place cursor on symbol
2. **Edit → Find Usages** (Alt+F7)
3. Review all locations

---

## Pre-Refactor Checklist

### ✅ Do These in PyCharm First

| Action | PyCharm Command | Notes |
|--------|-----------------|-------|
| Rename [network.py](file:///d:/Projects/sd-scripts/library/config/dataclasses/network.py) → [peft.py](file:///d:/Projects/sd-scripts/scripts/sd_peft.py) | Shift+F6 on file | Updates imports |
| Rename [NetworkConfig](file:///d:/Projects/sd-scripts/library/config/dataclasses/network.py#4-26) → [PeftConfig](file:///d:/Projects/sd-scripts/library/config/dataclasses/sd_peft.py#20-38) | Shift+F6 on class | Updates type hints |
| Rename [SDXLConfig](file:///d:/Projects/sd-scripts/library/config/dataclasses/sdxl.py#4-14) → (delete or merge) | Manual | Fields move elsewhere |
| Find all `cfg.network.` usages | Ctrl+Shift+F | Regex: `cfg\.network\.` |
| Find all `cfg.sdxl.` usages | Ctrl+Shift+F | Regex: `cfg\.sdxl\.` |

### ⚠️ Requires Manual/Script Work

| Task | Why PyCharm Can't Help |
|------|------------------------|
| YAML key renames | Not Python code |
| Nested restructuring (`cfg.X` → `cfg.Y.Z`) | Path change, not rename |
| Hydra defaults references | String-based |
| Documentation updates | Outside code index |

---

## Recommended PyCharm Workflow

### Step 1: Inventory Current State

```
Ctrl+Shift+F → Search: "cfg\.network\."
Ctrl+Shift+F → Search: "cfg\.sdxl\."
Ctrl+Shift+F → Search: "NetworkConfig"
```

Export results to a file for tracking.

### Step 2: Safe Renames First

1. **File renames** (won't break anything if imports update):
   - [network.py](file:///d:/Projects/sd-scripts/library/config/dataclasses/network.py) → [peft.py](file:///d:/Projects/sd-scripts/scripts/sd_peft.py)
   
2. **Class renames**:
   - [NetworkConfig](file:///d:/Projects/sd-scripts/library/config/dataclasses/network.py#4-26) → [PeftConfig](file:///d:/Projects/sd-scripts/library/config/dataclasses/sd_peft.py#20-38)

### Step 3: Structural Extraction

Create new nested dataclasses, e.g.:

```python
@dataclass
class LearningRatesConfig:
    base: float = 1e-4
    unet: Optional[float] = None
    text_encoders: Optional[Union[float, List[float]]] = None
```

Then add to [OptimizerConfig](file:///d:/Projects/sd-scripts/library/config/dataclasses/optimizer.py#4-24):
```python
learning_rates: LearningRatesConfig = field(default_factory=LearningRatesConfig)
```

### Step 4: Find-and-Replace Patterns

Use **Ctrl+Shift+R** (Replace in Path) with regex:

| Find (regex) | Replace |
|--------------|---------|
| `cfg\.network\.unet_lr` | `cfg.optimizer.learning_rates.unet` |
| `cfg\.network\.text_encoder_lr` | `cfg.optimizer.learning_rates.text_encoders` |
| `cfg\.sdxl\.learning_rate_te1` | `cfg.optimizer.learning_rates.text_encoders[0]` |

---

## Quick Reference: PyCharm Shortcuts

| Action | Shortcut |
|--------|----------|
| Rename | Shift+F6 |
| Find Usages | Alt+F7 |
| Find in Path | Ctrl+Shift+F |
| Replace in Path | Ctrl+Shift+R |
| Move (file/class) | F6 |
| Safe Delete | Alt+Delete |
| Undo Refactor | Ctrl+Z (with history) |

---

## What to Do After PyCharm Renames

1. **Run tests**: `pytest tests/unit/ -v`
2. **Check YAML files manually**: `configs/*.yaml`
3. **Grep for old names**: `grep -r "network_dim" .`
4. **Update YAML defaults** to match new structure
