from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn

from library.adapters.shared.trainables import AdapterTrainableParameterRef
from library.logging.summaries import (
    DiagnosticAlias,
    DiagnosticModuleCount,
    build_adapter_diagnostic_rows,
    build_trainer_diagnostic_rows,
    build_training_startup_summary,
    render_training_startup_summary,
)


def test_build_adapter_diagnostic_rows_uses_public_labels_and_internal_keys():
    adapter = SimpleNamespace()
    param_a = nn.Parameter(torch.ones(4))
    param_b = nn.Parameter(torch.ones(6))
    param_c = nn.Parameter(torch.ones(3), requires_grad=False)

    adapter.describe_trainable_parameter_refs = lambda: [
        AdapterTrainableParameterRef(
            param=param_a,
            name="lora_unet.block1.up",
            algorithm="lora",
            component="unet",
            component_key="denoiser",
            target_path="unet.block1",
            adapter_module_path="lora_unet.block1.lora_down",
        ),
        AdapterTrainableParameterRef(
            param=param_b,
            name="lora_unet.block1.up",
            algorithm="lora",
            component="unet",
            component_key="denoiser",
            target_path="unet.block1",
            adapter_module_path="lora_unet.block1.lora_up",
        ),
        AdapterTrainableParameterRef(
            param=param_c,
            name="lora_clip_l.block1.up",
            algorithm="lora",
            component="clip_l",
            component_key="text_encoder1",
            target_path="clip_l.block1",
            adapter_module_path="lora_clip_l.block1.lora_down",
        ),
    ]

    rows = build_adapter_diagnostic_rows(adapter)

    assert [(row.label, row.component_key) for row in rows] == [("unet", "denoiser"), ("clip_l", "text_encoder1")]
    assert rows[0].modules_total == 1
    assert rows[0].extra_module_counts == (DiagnosticModuleCount(label="adapter modules", trainable=2, total=2),)
    assert rows[0].params_trainable == 10
    assert rows[1].params_trainable == 0


def test_build_trainer_diagnostic_rows_maps_finetune_components_to_public_names():
    trainer = SimpleNamespace(
        adapter=None,
        cfg=SimpleNamespace(model=SimpleNamespace(model_type="sdxl")),
        text_encoders=[nn.Linear(2, 2), nn.Linear(2, 2)],
        mode=MagicMock(),
    )
    trainer.mode.get_diagnostics_components.return_value = (
        [("denoiser", nn.Linear(2, 2)), ("text_encoder1", nn.Linear(2, 2)), ("text_encoder2", nn.Linear(2, 2))],
        None,
    )

    rows, aliases = build_trainer_diagnostic_rows(trainer)

    assert aliases == []
    assert [row.label for row in rows] == ["unet", "clip_l", "clip_g"]
    assert [row.component_key for row in rows] == ["denoiser", "text_encoder1", "text_encoder2"]


def test_render_training_startup_summary_uses_sectioned_block_format():
    train_manifest = SimpleNamespace(entries={"a": SimpleNamespace(num_repeats=3, is_reg=False), "b": SimpleNamespace(num_repeats=1, is_reg=True)})
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(
            training=SimpleNamespace(train_batch_size=2, gradient_accumulation_steps=1),
            performance=SimpleNamespace(
                precision=SimpleNamespace(mixed_precision="bf16"),
                memory=SimpleNamespace(gradient_checkpointing=True),
                attention=SimpleNamespace(xformers=True),
                deepspeed=SimpleNamespace(deepspeed=False),
            ),
        ),
        accelerator=SimpleNamespace(num_processes=1),
        train_manifest=train_manifest,
        val_manifest=None,
        num_batches_per_epoch=10,
        num_train_epochs=1,
        max_train_steps=50,
        optimizer_name="AdamW8bit",
        optimizer=SimpleNamespace(param_groups=[]),
        optimization_plan=None,
        lr_descriptions=[],
        mode=SimpleNamespace(),
        strategies=SimpleNamespace(),
        adapter_method_name="lora",
    )
    component_rows = [
        SimpleNamespace(
            label="unet",
            component_key="denoiser",
            modules_trainable=2,
            modules_total=2,
            params_trainable=10,
            params_total=10,
            extra_module_counts=(),
            param_bytes_trainable=40,
            param_bytes_total=40,
        )
    ]
    summary = build_training_startup_summary(
        trainer=trainer,
        component_rows=component_rows,  # duck-typed DiagnosticRow fields
        aliases=[DiagnosticAlias(alias="trainable_model", target="unet")],
    )

    rendered = render_training_startup_summary(summary)

    assert "training run" in rendered
    assert "dataset" in rendered
    assert "schedule" in rendered
    assert "components" in rendered
    assert "optimizer" in rendered
    assert "method   : lora" in rendered
    assert "unet" in rendered
    assert "component |" in rendered
    assert "params" in rendered
    assert "trainable" in rendered
    assert "aliases: trainable_model -> unet" in rendered


def test_render_training_startup_summary_renders_extra_module_columns():
    summary = SimpleNamespace(
        runtime_rows=[],
        dataset_rows=[],
        schedule_rows=[],
        component_rows=[
            SimpleNamespace(
                label="unet",
                component_key="denoiser",
                modules_trainable=1,
                modules_total=1,
                params_trainable=10,
                params_total=10,
                extra_module_counts=(DiagnosticModuleCount(label="adapter modules", trainable=2, total=2),),
                param_bytes_trainable=40,
                param_bytes_total=40,
            )
        ],
        optimizer_name="AdamW8bit",
        optimizer_rows=[],
        aliases=[],
    )

    rendered = render_training_startup_summary(summary)

    assert "component |" in rendered
    assert "adapter modules" in rendered
    assert "1/1" in rendered
    assert "2/2" in rendered


def test_render_training_startup_summary_aligns_ratio_contents():
    summary = SimpleNamespace(
        runtime_rows=[],
        dataset_rows=[],
        schedule_rows=[],
        component_rows=[
            SimpleNamespace(
                label="clip_l",
                component_key="text_encoder1",
                modules_trainable=0,
                modules_total=99,
                params_trainable=0,
                params_total=123060480,
                extra_module_counts=(),
                param_bytes_trainable=0,
                param_bytes_total=0,
            ),
            SimpleNamespace(
                label="unet",
                component_key="denoiser",
                modules_trainable=1050,
                modules_total=1050,
                params_trainable=2567463684,
                params_total=2567463684,
                extra_module_counts=(),
                param_bytes_trainable=0,
                param_bytes_total=0,
            ),
        ],
        optimizer_name="AdamW8bit",
        optimizer_rows=[],
        aliases=[],
    )

    rendered = render_training_startup_summary(summary)

    assert "clip_l    |        0/  99 | 0/123,060,480" in rendered
    assert "unet      | 1,050/1,050 | 2,567,463,684/2,567,463,684" in rendered
