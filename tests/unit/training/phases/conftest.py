"""
Shared fixtures for training phase tests.

Provides mock_trainer and mock_cfg fixtures that simulate Trainer state
for unit testing phase functions in isolation.
"""

import pytest
from unittest.mock import MagicMock, PropertyMock
from types import SimpleNamespace

import torch


@pytest.fixture
def mock_cfg():
    """Create a deeply nested mock config matching the shared PEFT config structure."""
    cfg = MagicMock()

    # data.caching
    cfg.data.caching.cache_latents = True
    cfg.data.caching.cache_text_encoder_outputs = True
    cfg.data.caching.cache_text_encoder_outputs_to_disk = True
    cfg.data.caching.cache_dir = None
    cfg.data.caching.vae_batch_size = 1
    cfg.data.caching.te_batch_size = 1
    cfg.data.caching.num_workers = 0
    cfg.data.caching.cache_tokens_per_epoch = False

    # data.source
    cfg.data.source.train_data_dir = "/tmp/train"
    cfg.data.source.val_data_dir = None

    # data.preprocessing
    cfg.data.preprocessing.flip_aug = False

    # data.caption
    cfg.data.caption.shuffle_caption = False
    cfg.data.caption.keep_tokens = 0
    cfg.data.caption.caption_dropout_rate = 0.0
    cfg.data.caption.caption_tag_dropout_rate = 0.0
    cfg.data.caption.enable_wildcard = False
    cfg.data.caption.caption_separator = ","
    cfg.data.caption.secondary_separator = None
    cfg.data.caption.keep_tokens_separator = None
    cfg.data.caption.token_warmup_min = 1
    cfg.data.caption.token_warmup_step = 0

    # data.loader
    cfg.data.loader.num_workers = 0
    cfg.data.loader.prefetch_factor = 2
    cfg.data.loader.pin_memory = True
    cfg.data.loader.persistent_workers = False

    # performance.precision
    cfg.performance.precision.mixed_precision = "fp16"
    cfg.performance.precision.full_fp16 = False
    cfg.performance.precision.full_bf16 = False
    cfg.performance.precision.no_half_vae = False
    cfg.performance.precision.fp8_base = False
    cfg.performance.precision.fp8_base_unet = False

    # performance.memory
    cfg.performance.memory.offload_text_encoders = False
    cfg.performance.memory.lowram = False
    cfg.performance.memory.gradient_checkpointing = False
    cfg.performance.memory.load_denoiser_lazily = False

    # training
    cfg.training.seed = 42
    cfg.training.train_batch_size = 1
    cfg.training.max_train_epochs = 10
    cfg.training.max_train_steps = 1000
    cfg.training.max_token_length = 75
    cfg.training.gradient_accumulation_steps = 1

    # optimizer
    cfg.optimizer.optimizer_type = "AdamW"
    cfg.optimizer.max_grad_norm = 1.0
    cfg.optimizer.learning_rates = MagicMock()  # Nested learning rates config

    # peft
    cfg.peft.adapter_module = "library.adapters.lora"
    cfg.peft.adapter_rank = 4
    cfg.peft.adapter_alpha = 1.0
    cfg.peft.adapter_weights = None
    cfg.peft.adapter_rank_from_weights = False
    cfg.peft.adapter_args = None
    cfg.peft.base_weights = None
    cfg.peft.base_weights_multiplier = None
    cfg.peft.neuron_dropout = 0.0
    cfg.peft.scale_weight_norms = False

    # output.saving
    cfg.output.saving.output_dir = "/tmp/output"
    cfg.output.saving.output_name = "test_model"
    cfg.output.saving.save_model_as = "safetensors"
    cfg.output.saving.save_every_n_steps = None
    cfg.output.saving.save_every_n_epochs = None
    cfg.output.saving.save_state = False
    cfg.output.saving.save_n_epoch_ratio = None

    # output.sampling
    cfg.output.sampling.sample_every_n_steps = None
    cfg.output.sampling.sample_every_n_epochs = None

    # output.logging
    cfg.output.logging.log_timestep_distribution_every_n_steps = None

    # loss
    cfg.loss.prior_loss_weight = 1.0
    cfg.loss.edm2.edm2_loss_weighting = False

    # validation
    cfg.validation.validate_every_n_steps = None
    cfg.validation.validation_seed = 42

    # performance (additional)
    cfg.performance.deepspeed = False

    # output.huggingface
    cfg.output.huggingface = None

    return cfg


@pytest.fixture
def mock_accelerator():
    """Create a mock Accelerator with common method stubs."""
    accelerator = MagicMock()
    accelerator.device = torch.device("cpu")
    accelerator.process_index = 0
    accelerator.num_processes = 1
    accelerator.is_main_process = True
    accelerator.sync_gradients = True
    accelerator.gradient_accumulation_steps = 1

    # Methods
    accelerator.wait_for_everyone = MagicMock()
    accelerator.unwrap_model = MagicMock(side_effect=lambda x: x)
    accelerator.backward = MagicMock()
    accelerator.clip_grad_norm_ = MagicMock()
    accelerator.prepare = MagicMock(side_effect=lambda *args: args if len(args) > 1 else args[0])
    accelerator.print = MagicMock()
    accelerator.log = MagicMock()

    # accumulate context manager
    accelerator.accumulate = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

    return accelerator


@pytest.fixture
def mock_strategies():
    """Create a mock training strategies object."""
    strategies = MagicMock()
    strategies.cast_denoiser = MagicMock(return_value=True)
    strategies.cast_text_encoder = MagicMock(return_value=True)
    strategies.post_process_trainable = MagicMock()
    strategies.prepare_text_encoder_fp8 = MagicMock()
    strategies.load_denoiser_lazily = MagicMock(return_value=(MagicMock(), []))
    strategies.get_token_cache_encoder_names = MagicMock(return_value=["clip_l", "clip_g"])
    strategies.build_te_cache_model_bundle = MagicMock(return_value=("te1", "te2", "tok1", "tok2"))
    strategies.process_batch = MagicMock(return_value=(torch.tensor(0.5), torch.tensor(0.5), None, torch.tensor([500])))
    strategies.all_reduce_trainable = MagicMock()
    strategies.sample_images = MagicMock()
    strategies.calculate_val_loss = MagicMock(return_value=(None, None))
    strategies.la_sampler = None
    strategies.live_plotter_process = None
    return strategies


@pytest.fixture
def mock_trainer(mock_cfg, mock_accelerator, mock_strategies):
    """Create a mock Trainer with all required attributes for phase testing."""
    trainer = MagicMock()

    # Core config and dependencies
    trainer.cfg = mock_cfg
    trainer._accelerator = mock_accelerator
    trainer.strategies = mock_strategies

    # Accelerator property simulation
    type(trainer).accelerator = PropertyMock(return_value=mock_accelerator)
    type(trainer).is_main_process = PropertyMock(return_value=True)

    # Models (all mocked)
    trainer.vae = MagicMock()
    trainer.vae.to = MagicMock(return_value=trainer.vae)
    trainer.vae.requires_grad_ = MagicMock()
    trainer.vae.eval = MagicMock()

    trainer.denoiser = MagicMock()
    trainer.denoiser.to = MagicMock(return_value=trainer.denoiser)
    trainer.denoiser.requires_grad_ = MagicMock()
    trainer.denoiser.enable_gradient_checkpointing = MagicMock()

    trainer.text_encoders = [MagicMock(), MagicMock()]
    for te in trainer.text_encoders:
        te.to = MagicMock(return_value=te)
        te.requires_grad_ = MagicMock()
        te.eval = MagicMock()
        te.device = MagicMock()
        te.device.type = "cuda"
        te.gradient_checkpointing_enable = MagicMock()

    trainer._text_encoder = trainer.text_encoders
    trainer.tokenizers = [MagicMock(), MagicMock()]

    # Dtypes
    trainer.weight_dtype = torch.float16
    trainer.vae_dtype = torch.float32
    trainer.denoiser_weight_dtype = torch.float16
    trainer.te_weight_dtype = torch.float16

    # Manifests
    trainer.train_manifest = MagicMock()
    trainer.train_manifest.entries = {}
    trainer.val_manifest = None

    # Strategies (cached versions)
    trainer.latent_cache_backend = None
    trainer.te_cache_backend = None
    # Tokenization and text encoding now live directly on TrainingStrategy.

    # Adapter
    trainer.adapter = MagicMock()
    trainer.adapter.to = MagicMock()
    trainer.adapter.apply_to = MagicMock()
    trainer.adapter.get_trainable_params = MagicMock(return_value=[torch.nn.Parameter(torch.randn(10))])
    trainer.adapter.on_epoch_start = MagicMock()
    trainer.adapter.eval = MagicMock()
    trainer.adapter.train = MagicMock()
    trainer.net_kwargs = {}

    # Training mode (TrainingMode protocol)
    trainer.mode = MagicMock()
    trainer.mode.on_epoch_start = MagicMock()
    trainer.mode.on_step_start = MagicMock()
    trainer.mode.on_step_end = MagicMock(return_value={})
    trainer.mode.get_trainable_params = MagicMock(return_value=[torch.nn.Parameter(torch.randn(10))])
    trainer.mode.set_eval = MagicMock()
    trainer.mode.set_train = MagicMock()
    trainer.mode.save_checkpoint = MagicMock()

    # Training flags
    trainer._train_denoiser = True
    trainer._train_text_encoder = False

    # Optimizer/Scheduler
    trainer.optimizer = MagicMock()
    trainer.optimizer.step = MagicMock()
    trainer.optimizer.zero_grad = MagicMock()
    trainer.lr_scheduler = MagicMock()
    trainer.lr_scheduler.step = MagicMock()
    trainer.lr_descriptions = ["denoiser"]
    trainer.optimizer_train_fn = MagicMock()
    trainer.optimizer_eval_fn = MagicMock()

    # Training state
    trainer.global_step = 0
    trainer.epoch_to_start = 0
    trainer.num_train_epochs = 1
    trainer.max_train_steps = 100
    trainer.num_batches_per_epoch = 10
    trainer._initial_step = 0
    trainer._accumulation_counter = 0

    # Epoch/step state containers
    trainer._current_epoch_state = SimpleNamespace(value=0)
    trainer._current_step_state = SimpleNamespace(value=0)

    # Loss tracking
    trainer._loss_recorder = MagicMock()
    trainer._loss_recorder.add = MagicMock()
    trainer._loss_recorder.average = 0.5
    trainer._val_loss_recorder = None
    trainer._loss_scaled_recorder = None
    trainer._current_global_step_loss = 0.0
    trainer._current_global_step_loss_scaled = None
    trainer._current_val_loss = None
    trainer._average_val_loss = None

    # Progress bar
    trainer._progress_bar = MagicMock()
    trainer._progress_bar.update = MagicMock()
    trainer._progress_bar.set_postfix = MagicMock()
    trainer._progress_bar.unpause = MagicMock()

    # Misc state
    trainer._is_tracking = False
    trainer._metadata = {}
    trainer._cache_dir = "/tmp/cache"
    trainer._n_workers = 0
    trainer._val_dataloader = None
    trainer._cyclic_val_dataloader = None
    trainer._training_model = trainer.adapter
    trainer._edm2_model = None
    trainer._edm2_optimizer = None
    trainer._edm2_lr_scheduler = None
    trainer._timestep_counts = None
    trainer._plotter_settings = None
    trainer._dynamic_timestep_schedule = []
    trainer._current_min_timestep = 0
    trainer._current_max_timestep = 1000

    # Validation scheduler (always returns False for should_run by default)
    trainer._validation_scheduler = MagicMock()
    trainer._validation_scheduler.should_run = MagicMock(return_value=False)
    trainer._resource_monitor = MagicMock()

    # trainable_model property (returns adapter for PEFT)
    type(trainer).trainable_model = PropertyMock(return_value=trainer.adapter)

    # Noise scheduler
    trainer.noise_scheduler = MagicMock()

    # Methods
    trainer.save_checkpoint = MagicMock()
    trainer.remove_checkpoint = MagicMock()

    return trainer
