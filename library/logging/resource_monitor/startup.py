from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from library.logging.summaries import DiagnosticRow
from library.metadata.dataclasses.resource import StructuralResourceFacts
from library.metadata.records import MetadataValue


_MIB = 1024 * 1024
_STARTUP_TRAINING_STATE_ESTIMATOR_VERSION = "startup_training_state_v1"


@dataclass(frozen=True)
class _StartupComponentMemory:
    label: str
    component_key: str
    param_bytes_total: int
    param_bytes_trainable: int


@dataclass(frozen=True)
class _StartupTrainingStateEstimate:
    total_param_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: float
    optimizer_name: str
    optimizer_state_multiplier: float
    optimizer_state_multiplier_source: str
    optimizer_state_basis: str
    component_count: int
    deepspeed_enabled: bool
    deepspeed_zero_stage: int | None
    caveats: tuple[str, ...]


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
                        component_key=item.component_key,
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
                    component_key=name,
                    param_bytes_total=param_bytes_total,
                    param_bytes_trainable=param_bytes_trainable,
                )
            )

        return startup_rows

    def _build_startup_structural_facts(
        self,
        startup_rows: Iterable[_StartupComponentMemory],
        *,
        run_identifier: str,
        training_state_estimate: _StartupTrainingStateEstimate | None = None,
    ) -> tuple[StructuralResourceFacts, ...]:
        facts: list[StructuralResourceFacts] = []
        for row in startup_rows:
            metadata = {
                "display_label": row.label,
                "param_bytes_total": row.param_bytes_total,
                "param_bytes_trainable": row.param_bytes_trainable,
            }
            facts.append(
                StructuralResourceFacts(
                    structural_identifier=f"{run_identifier}:startup_component:{row.component_key}:parameter_memory",
                    run_identifier=run_identifier,
                    owner_type="model_component",
                    owner_identifier=row.component_key,
                    resource_kind="parameter_memory",
                    quantity=row.param_bytes_total / (1024 * 1024),
                    unit="MiB",
                    basis="parameter_bytes",
                    source="startup_component_memory",
                    component_key=row.component_key,
                    metadata=metadata,
                )
            )
            facts.append(
                StructuralResourceFacts(
                    structural_identifier=f"{run_identifier}:startup_component:{row.component_key}:trainable_parameter_memory",
                    run_identifier=run_identifier,
                    owner_type="model_component",
                    owner_identifier=row.component_key,
                    resource_kind="trainable_parameter_memory",
                    quantity=row.param_bytes_trainable / (1024 * 1024),
                    unit="MiB",
                    basis="trainable_parameter_bytes",
                    source="startup_component_memory",
                    component_key=row.component_key,
                    metadata=metadata,
                )
            )
        if training_state_estimate is not None:
            facts.extend(
                self._build_startup_training_state_structural_facts(
                    training_state_estimate,
                    run_identifier=run_identifier,
                )
            )
        return tuple(facts)

    def _build_startup_training_state_structural_facts(
        self,
        estimate: _StartupTrainingStateEstimate,
        *,
        run_identifier: str,
    ) -> tuple[StructuralResourceFacts, ...]:
        shared_metadata = self._startup_training_state_metadata(estimate)
        return (
            StructuralResourceFacts(
                structural_identifier=f"{run_identifier}:startup_training_state:gradients:gradient_memory",
                run_identifier=run_identifier,
                owner_type="training_state",
                owner_identifier="gradients",
                resource_kind="gradient_memory",
                quantity=estimate.gradient_bytes / _MIB,
                unit="MiB",
                basis="trainable_parameter_bytes_estimate",
                source="startup_training_state_estimator",
                group_identifier="training_state",
                validity_scope="startup_estimate",
                metadata={
                    **shared_metadata,
                    "quantity_bytes": estimate.gradient_bytes,
                    "basis_detail": "sum(param_bytes_trainable)",
                },
            ),
            StructuralResourceFacts(
                structural_identifier=f"{run_identifier}:startup_training_state:optimizer_state:optimizer_state_memory",
                run_identifier=run_identifier,
                owner_type="training_state",
                owner_identifier="optimizer_state",
                resource_kind="optimizer_state_memory",
                quantity=estimate.optimizer_state_bytes / _MIB,
                unit="MiB",
                basis=estimate.optimizer_state_basis,
                source="startup_training_state_estimator",
                group_identifier="training_state",
                validity_scope="startup_estimate",
                metadata={
                    **shared_metadata,
                    "quantity_bytes": estimate.optimizer_state_bytes,
                    "basis_detail": "sum(param_bytes_trainable) * optimizer_state_multiplier",
                },
            ),
        )

    def _startup_training_state_metadata(
        self,
        estimate: _StartupTrainingStateEstimate,
    ) -> dict[str, MetadataValue]:
        metadata: dict[str, MetadataValue] = {
            "estimator_version": _STARTUP_TRAINING_STATE_ESTIMATOR_VERSION,
            "component_count": estimate.component_count,
            "optimizer_name": estimate.optimizer_name,
            "optimizer_state_multiplier": estimate.optimizer_state_multiplier,
            "optimizer_state_multiplier_source": estimate.optimizer_state_multiplier_source,
            "total_param_bytes": estimate.total_param_bytes,
            "total_trainable_bytes": estimate.gradient_bytes,
            "deepspeed_enabled": estimate.deepspeed_enabled,
        }
        if estimate.deepspeed_zero_stage is not None:
            metadata["deepspeed_zero_stage"] = estimate.deepspeed_zero_stage
        if estimate.caveats:
            metadata["caveats"] = list(estimate.caveats)
        return metadata

    def _file_startup_structural_facts(
        self,
        startup_rows: Iterable[_StartupComponentMemory],
        *,
        training_state_estimate: _StartupTrainingStateEstimate | None = None,
    ) -> None:
        metadata_runtime = getattr(self, "_metadata_runtime", None)
        run_identifier = getattr(self, "_run_identifier", None)
        if metadata_runtime is None or run_identifier is None:
            return

        facts = self._build_startup_structural_facts(
            startup_rows,
            run_identifier=run_identifier,
            training_state_estimate=training_state_estimate,
        )
        if facts:
            metadata_runtime.file_many(facts)

    def _estimate_startup_training_state(
        self,
        startup_rows: Iterable[_StartupComponentMemory],
        *,
        optimizer_name: str,
        deepspeed_enabled: bool,
        deepspeed_zero_stage: int | None,
        optimizer_state_multiplier: float | None,
    ) -> _StartupTrainingStateEstimate:
        rows = tuple(startup_rows)
        total_param_bytes = sum(row.param_bytes_total for row in rows)
        total_trainable_bytes = sum(row.param_bytes_trainable for row in rows)
        multiplier, multiplier_source, optimizer_state_basis, caveats = self._estimate_optimizer_state_multiplier(
            optimizer_name,
            optimizer_state_multiplier=optimizer_state_multiplier,
        )
        if deepspeed_enabled:
            caveats = (
                *caveats,
                "deepspeed_zero_partitioning_or_offload_may_change_per_rank_footprint",
            )
        return _StartupTrainingStateEstimate(
            total_param_bytes=total_param_bytes,
            gradient_bytes=total_trainable_bytes,
            optimizer_state_bytes=total_trainable_bytes * multiplier,
            optimizer_name=optimizer_name,
            optimizer_state_multiplier=multiplier,
            optimizer_state_multiplier_source=multiplier_source,
            optimizer_state_basis=optimizer_state_basis,
            component_count=len(rows),
            deepspeed_enabled=deepspeed_enabled,
            deepspeed_zero_stage=deepspeed_zero_stage,
            caveats=caveats,
        )

    def _estimate_optimizer_state_multiplier(
        self,
        optimizer_name: str,
        *,
        optimizer_state_multiplier: float | None,
    ) -> tuple[float, str, str, tuple[str, ...]]:
        if optimizer_state_multiplier is not None:
            multiplier = float(optimizer_state_multiplier)
            caveats = ("negative_optimizer_state_multiplier_clamped",) if multiplier < 0 else ()
            return (
                max(multiplier, 0.0),
                "domain_provided_multiplier",
                "domain_provided_optimizer_state_multiplier",
                caveats,
            )

        normalized = optimizer_name.lower()
        if "adam" in normalized or "lion" in normalized:
            return (
                2.0,
                "optimizer_name_heuristic",
                "two_state_optimizer_heuristic",
                ("optimizer_state_layout_not_observed",),
            )
        return (
            1.0,
            "optimizer_name_heuristic",
            "fallback_single_state_optimizer_heuristic",
            ("optimizer_state_layout_not_observed",),
        )

    def emit_startup_component_memory(
        self,
        components: Mapping[str, Any] | Iterable[tuple[str, Any]] | Iterable[DiagnosticRow] | None,
        optimizer_name: str,
        *,
        deepspeed_enabled: bool = False,
        deepspeed_zero_stage: int | None = None,
        optimizer_state_multiplier: float | None = None,
    ) -> None:
        """Log startup memory and file structural facts when metadata is available.

        ``optimizer_state_multiplier`` is an optional owner-provided estimate
        seam. When omitted, startup output preserves the legacy optimizer-name
        heuristic and records that limitation in structural fact metadata.
        """
        try:
            if not self._should_emit_this_rank():
                return
            if not self._component_breakdown:
                return
            startup_rows = self._collect_startup_component_memory(components)
            if not startup_rows:
                return
            training_state_estimate = self._estimate_startup_training_state(
                startup_rows,
                optimizer_name=optimizer_name,
                deepspeed_enabled=deepspeed_enabled,
                deepspeed_zero_stage=deepspeed_zero_stage,
                optimizer_state_multiplier=optimizer_state_multiplier,
            )
            self._file_startup_structural_facts(
                startup_rows,
                training_state_estimate=training_state_estimate,
            )

            lines = ["Resource startup breakdown:"]
            total_param_bytes = training_state_estimate.total_param_bytes
            total_trainable_bytes = training_state_estimate.gradient_bytes
            lines.append("  loaded model weights:")
            total_frozen_bytes = max(total_param_bytes - total_trainable_bytes, 0)

            table_rows: list[tuple[str, str, str, str, str]] = []
            for row in startup_rows:
                loaded_mb = row.param_bytes_total / _MIB
                trainable_mb = row.param_bytes_trainable / _MIB
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
                    f"{total_param_bytes / _MIB:.1f}MB",
                    f"{total_trainable_bytes / _MIB:.1f}MB",
                    f"{total_frozen_bytes / _MIB:.1f}MB",
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

            optimizer_state_bytes = training_state_estimate.optimizer_state_bytes
            lines.append("  training state:")
            lines.append(f"    - gradients (est): {total_trainable_bytes / _MIB:.1f}MB")
            lines.append(f"    - optimizer_state (est): {optimizer_state_bytes / _MIB:.1f}MB [{optimizer_name}]")
            lines.append(f"    - total (est): {(total_param_bytes + total_trainable_bytes + optimizer_state_bytes) / _MIB:.1f}MB")
            if deepspeed_enabled:
                zero_stage_label = "n/a" if deepspeed_zero_stage is None else str(deepspeed_zero_stage)
                lines.append(
                    "    - DeepSpeed/ZeRO caveat: effective per-rank footprint may be lower/higher than these estimates "
                    f"due to state partitioning/offload (zero_stage={zero_stage_label})."
                )
            self._print_block_external("\n" + "\n".join(lines))
        except Exception as exc:  # pragma: no cover - defensive safety net
            self._warn_once("startup_component_memory", "resource monitor startup component estimate failed: %s", exc)
