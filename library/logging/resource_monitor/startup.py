from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from library.logging.summaries import DiagnosticRow


@dataclass(frozen=True)
class _StartupComponentMemory:
    label: str
    param_bytes_total: int
    param_bytes_trainable: int


class ResourceStartupMixin:
    def _normalize_component_items(
        self,
        components: Mapping[str, Any] | Iterable[tuple[str, Any]] | Iterable[DiagnosticRow] | None,
    ) -> list[tuple[str, Any] | DiagnosticRow]:
        if not components:
            return []

        if isinstance(components, Mapping):
            raw_items = list(components.items())
        else:
            raw_items = list(components)

        normalized: list[tuple[str, Any] | DiagnosticRow] = []
        for item in raw_items:
            if isinstance(item, DiagnosticRow):
                normalized.append(item)
                continue
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            name, module = item
            if not isinstance(name, str) or module is None:
                continue
            normalized.append((name, module))
        return normalized

    def _collect_startup_component_memory(
        self,
        components: Mapping[str, Any] | Iterable[tuple[str, Any]] | Iterable[DiagnosticRow] | None,
    ) -> list[_StartupComponentMemory]:
        component_items = self._normalize_component_items(components)
        startup_rows: list[_StartupComponentMemory] = []

        for item in component_items:
            if isinstance(item, DiagnosticRow):
                startup_rows.append(
                    _StartupComponentMemory(
                        label=item.label,
                        param_bytes_total=item.param_bytes_total,
                        param_bytes_trainable=item.param_bytes_trainable,
                    )
                )
                continue

            name, module = item
            param_bytes_total = 0
            param_bytes_trainable = 0
            for p in module.parameters():
                bytes_count = p.numel() * p.element_size()
                param_bytes_total += bytes_count
                if p.requires_grad:
                    param_bytes_trainable += bytes_count
            startup_rows.append(
                _StartupComponentMemory(
                    label=name,
                    param_bytes_total=param_bytes_total,
                    param_bytes_trainable=param_bytes_trainable,
                )
            )

        return startup_rows

    def emit_startup_component_memory(
        self,
        components: Mapping[str, Any] | Iterable[tuple[str, Any]] | Iterable[DiagnosticRow] | None,
        optimizer_name: str,
        *,
        deepspeed_enabled: bool = False,
        deepspeed_zero_stage: int | None = None,
    ) -> None:
        try:
            if not self._should_emit_this_rank():
                return
            if not self._component_breakdown:
                return
            startup_rows = self._collect_startup_component_memory(components)
            if not startup_rows:
                return

            lines = ["Resource startup breakdown:"]
            total_param_bytes = sum(row.param_bytes_total for row in startup_rows)
            total_trainable_bytes = sum(row.param_bytes_trainable for row in startup_rows)
            lines.append("  loaded model weights:")
            total_frozen_bytes = max(total_param_bytes - total_trainable_bytes, 0)

            table_rows: list[tuple[str, str, str, str, str]] = []
            for row in startup_rows:
                loaded_mb = row.param_bytes_total / (1024 * 1024)
                trainable_mb = row.param_bytes_trainable / (1024 * 1024)
                frozen_mb = max(loaded_mb - trainable_mb, 0.0)
                share_percent = (row.param_bytes_total / total_param_bytes * 100) if total_param_bytes > 0 else 0.0
                table_rows.append(
                    (
                        row.label,
                        f"{loaded_mb:.1f}MB",
                        f"{trainable_mb:.1f}MB",
                        f"{frozen_mb:.1f}MB",
                        f"{share_percent:.1f}%",
                    )
                )

            table_rows.append(
                (
                    "total",
                    f"{total_param_bytes / (1024 * 1024):.1f}MB",
                    f"{total_trainable_bytes / (1024 * 1024):.1f}MB",
                    f"{total_frozen_bytes / (1024 * 1024):.1f}MB",
                    "100.0%",
                )
            )

            if not table_rows:
                return

            component_width = max(len("component"), *(len(row[0]) for row in table_rows))
            loaded_width = max(len("loaded"), *(len(row[1]) for row in table_rows))
            trainable_width = max(len("trainable"), *(len(row[2]) for row in table_rows))
            frozen_width = max(len("frozen"), *(len(row[3]) for row in table_rows))
            share_width = max(len("share"), *(len(row[4]) for row in table_rows))

            lines.append(
                "    "
                f"{'component':<{component_width}} | "
                f"{'loaded':>{loaded_width}} | "
                f"{'trainable':>{trainable_width}} | "
                f"{'frozen':>{frozen_width}} | "
                f"{'share':>{share_width}}"
            )
            lines.append(
                "    "
                f"{'-' * component_width}-+-"
                f"{'-' * loaded_width}-+-"
                f"{'-' * trainable_width}-+-"
                f"{'-' * frozen_width}-+-"
                f"{'-' * share_width}"
            )
            for component, loaded, trainable, frozen, share in table_rows:
                lines.append(
                    "    "
                    f"{component:<{component_width}} | "
                    f"{loaded:>{loaded_width}} | "
                    f"{trainable:>{trainable_width}} | "
                    f"{frozen:>{frozen_width}} | "
                    f"{share:>{share_width}}"
                )

            optimizer_state_multiplier = 2 if "adam" in optimizer_name.lower() or "lion" in optimizer_name.lower() else 1
            optimizer_state_bytes = total_trainable_bytes * optimizer_state_multiplier
            lines.append("  training state:")
            lines.append(f"    - gradients (est): {total_trainable_bytes / (1024 * 1024):.1f}MB")
            lines.append(f"    - optimizer_state (est): {optimizer_state_bytes / (1024 * 1024):.1f}MB [{optimizer_name}]")
            lines.append(f"    - total (est): {(total_param_bytes + total_trainable_bytes + optimizer_state_bytes) / (1024 * 1024):.1f}MB")
            if deepspeed_enabled:
                zero_stage_label = "n/a" if deepspeed_zero_stage is None else str(deepspeed_zero_stage)
                lines.append(
                    "    - DeepSpeed/ZeRO caveat: effective per-rank footprint may be lower/higher than these estimates "
                    f"due to state partitioning/offload (zero_stage={zero_stage_label})."
                )
            self._print_block_external("\n" + "\n".join(lines))
        except Exception as exc:  # pragma: no cover - defensive safety net
            self._warn_once("startup_component_memory", "resource monitor startup component estimate failed: %s", exc)
