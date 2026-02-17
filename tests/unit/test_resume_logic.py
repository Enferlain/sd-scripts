import unittest
from unittest.mock import MagicMock
import sys
import os
import json
from types import SimpleNamespace
import tempfile
import shutil

# Add repo root to path
sys.path.append(os.getcwd())

from library.training.runners.peft_trainer import PeftTrainer
from library.training.phases.optimizer import prepare_optimizer


class MockAccelerator:
    def __init__(self):
        self.state = SimpleNamespace(epoch=SimpleNamespace(value=0), step=SimpleNamespace(value=0))
        self.process_index = 0
        self.num_processes = 1
        self.device = "cpu"
        self.is_main_process = True
        self.is_local_main_process = True
        self.trackers = []
        self._save_hooks = []
        self._load_hooks = []

    def register_save_state_pre_hook(self, hook):
        self._save_hooks.append(hook)

    def register_load_state_pre_hook(self, hook):
        self._load_hooks.append(hook)

    def print(self, *args, **kwargs):
        pass

    def prepare(self, *args):
        return args

    def unwrap_model(self, model):
        return model

    def load_state(self, input_dir):
        # Simulate Accelerate's load_state triggering hooks
        models = []  # Mock models list
        for hook in self._load_hooks:
            hook(models, input_dir)


class MockStrategy:
    def is_train_unet(self, cfg):
        return True

    def is_train_text_encoder(self, cfg):
        return False

    def get_text_encoders_train_flags(self, cfg, te):
        return [False] * len(te)

    def cast_unet(self, cfg):
        return False

    def cast_vae(self, cfg):
        return False

    def cast_text_encoder(self, cfg):
        return False

    def load_target_model(self, *args):
        return "v1", [], MagicMock(), MagicMock()

    def get_tokenize_strategy(self, cfg):
        return MagicMock()

    def get_tokenizers(self, strategy):
        return []

    def get_latents_caching_strategy(self, cfg):
        return MagicMock()

    def prepare_unet_with_accelerator(self, cfg, accelerator, unet):
        return unet


class TestResumeBehavior(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.cfg = MagicMock()
        self.cfg.optimizer = MagicMock()
        self.cfg.optimizer.learning_rates = []
        self.cfg.peft = MagicMock()
        self.cfg.data = MagicMock()
        self.cfg.data.loader.num_workers = 0
        self.cfg.training.max_train_epochs = 1
        self.cfg.training.max_train_steps = 100
        self.cfg.training.gradient_accumulation_steps = 1
        self.cfg.training.train_batch_size = 1
        self.cfg.output.saving.save_n_epoch_ratio = None
        self.cfg.performance.precision.full_fp16 = False
        self.cfg.performance.deepspeed = None
        self.cfg.performance.memory.gradient_checkpointing = False
        self.cfg.loss.prior_loss_weight = 1.0
        self.cfg.validation.validation_seed = 42
        self.cfg.data.loader.prefetch_factor = None
        self.cfg.data.loader.pin_memory = False
        self.cfg.data.loader.persistent_workers = False

        self.cfg.output.saving.resume = None
        self.cfg.output.huggingface.resume_from_huggingface = False

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_resume_logic(self):
        # 1. SETUP TRAINER & MOCKING
        strategy = MockStrategy()
        mode = MagicMock()
        # Configure mode mocks for prepare_optimizer
        mode.build_optimizer_params.return_value = ("AdamW", {}, MagicMock(), MagicMock(), MagicMock(), [])
        trainer = PeftTrainer(self.cfg, strategy, mode)
        trainer._accelerator = MockAccelerator()  # Use backing field since accelerator is a property
        trainer.adapter = MagicMock()  # Mock adapter
        trainer.val_manifest = None  # No validation for this test

        # Manually set setup() outputs
        trainer.train_manifest = MagicMock()
        # Mock entries in manifest to avoid zero-division or empty loops
        trainer.train_manifest.entries = {"img1": MagicMock(num_repeats=1, is_reg=False)}
        trainer.num_batches_per_epoch = 10
        trainer.text_encoders = [MagicMock()]
        trainer.unet = MagicMock()
        trainer.vae = MagicMock()

        # 2. SIMULATE TRAINING STATE (Step 10, Epoch 1)
        trainer.global_step = 10
        trainer.current_epoch = 1
        trainer._current_step_state.value = 10
        trainer._current_epoch_state.value = 1

        # 3. TRIGGER SAVE HOOK
        # We need to call register_adapter_state_hooks to get the hooks registered
        from library.training.checkpointing import register_adapter_state_hooks

        _ = register_adapter_state_hooks(
            trainer.accelerator, trainer.adapter, self.cfg, trainer._current_epoch_state, trainer._current_step_state
        )

        save_dir = os.path.join(self.test_dir, "checkpoint-10")
        os.makedirs(save_dir, exist_ok=True)

        # Execute the registered save hook
        save_hook = trainer.accelerator._save_hooks[0]
        save_hook([], [], save_dir)

        # Verify json exists and content
        json_path = os.path.join(save_dir, "train_state.json")
        self.assertTrue(os.path.exists(json_path), "train_state.json should exist")
        with open(json_path) as f:
            state = json.load(f)
            # Note: Hook saves current_step + 1
            print(f"Saved State: {state}")
            self.assertEqual(state["current_step"], 11)
            self.assertEqual(state["current_epoch"], 1)

        # 4. SIMULATE RESUME
        # Reset trainer "memory"
        trainer.global_step = 0
        trainer.epoch_to_start = 0
        trainer._initial_step = 0
        trainer._current_step_state.value = 0
        trainer._current_epoch_state.value = 0

        # Set config to resume
        self.cfg.output.saving.resume = save_dir

        # Register hooks again (as done in prepare_optimizer)
        # Note: We must clear hooks from mock accelerator first to avoid duplication/confusion
        trainer.accelerator._save_hooks = []
        trainer.accelerator._load_hooks = []

        # Run the prepare_optimizer function (which contains the logic we are testing)
        # Configure mode.register_state_hooks to use the real register_adapter_state_hooks
        from library.training.checkpointing import register_adapter_state_hooks as real_register

        def mock_register_state_hooks(t):
            return real_register(t.accelerator, t.adapter, self.cfg, t._current_epoch_state, t._current_step_state)

        trainer.mode.register_state_hooks.side_effect = mock_register_state_hooks

        with (
            unittest.mock.patch("library.training.phases.optimizer.get_scheduler_fix") as mock_sched,
            unittest.mock.patch("library.training.phases.optimizer._setup_gradient_checkpointing") as mock_grad,
        ):
            # This is the function under test!
            prepare_optimizer(trainer)

            print(f"Trainer Global Step after resume: {trainer.global_step}")
            print(f"Trainer Initial Step after resume: {trainer._initial_step}")
            print(f"Trainer Epoch to Start after resume: {trainer.epoch_to_start}")
            print(f"Trainer Current Step State after resume: {trainer._current_step_state.value}")
            print(f"Trainer Current Epoch State after resume: {trainer._current_epoch_state.value}")

            # The Critical Checks
            self.assertEqual(trainer._initial_step, 11, "Initial step should be 11")
            self.assertEqual(trainer.epoch_to_start, 1, "Epoch to start should be 1")

            # FIXED BEHAVIOR CHECKS
            self.assertEqual(trainer.global_step, 11, "Global step should be 11 (resumed)")
            self.assertEqual(trainer._current_step_state.value, 11, "Current step state should be updated to 11")
            self.assertEqual(trainer._current_epoch_state.value, 1, "Current epoch state should be updated to 1")


if __name__ == "__main__":
    unittest.main()
