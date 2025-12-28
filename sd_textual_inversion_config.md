### DATA ###
----------------------------------------

  cfg.data (2 occurrences)
    L 243: data_config = cfg.data
    L 473: cfg.data,

### LOSS ###
----------------------------------------

  cfg.loss (1 occurrences)
    L 730: cfg.loss, timesteps, noise_scheduler

  cfg.loss.debiased_estimation_loss (1 occurrences)
    L 768: if cfg.loss.debiased_estimation_loss:

  cfg.loss.loss_scale (1 occurrences)
    L 738: scale=float(cfg.loss.loss_scale),

  cfg.loss.loss_type (1 occurrences)
    L 735: cfg.loss.loss_type,

  cfg.loss.min_snr_gamma (2 occurrences)
    L 749: if cfg.loss.min_snr_gamma:
    L 754: cfg.loss.min_snr_gamma,

  cfg.loss.regularization (1 occurrences)
    L 703: cfg.loss.regularization,

  cfg.loss.scale_v_pred_loss_like_noise_pred (1 occurrences)
    L 757: if cfg.loss.scale_v_pred_loss_like_noise_pred:

  cfg.loss.v_parameterization (4 occurrences)
    L 620: v_parameterization=cfg.loss.v_parameterization,
    L 724: if cfg.loss.v_parameterization:
    L 755: cfg.loss.v_parameterization,
    L 773: cfg.loss.v_parameterization,

  cfg.loss.v_pred_like_loss (2 occurrences)
    L 761: if cfg.loss.v_pred_like_loss:
    L 766: cfg.loss.v_pred_like_loss,

### MASKED_LOSS ###
----------------------------------------

  cfg.masked_loss.masked_loss (1 occurrences)
    L 740: if cfg.masked_loss.masked_loss or (

### MODEL ###
----------------------------------------

  cfg.model (1 occurrences)
    L 244: model_config = cfg.model

### OPTIMIZER ###
----------------------------------------

  cfg.optimizer (2 occurrences)
    L 245: optimizer_config = cfg.optimizer
    L 472: cfg.optimizer,

### OUTPUT ###
----------------------------------------

  cfg.output.huggingface (1 occurrences)
    L 628: cfg.output.huggingface,

  cfg.output.huggingface.huggingface_repo_id (1 occurrences)
    L 626: if cfg.output.huggingface.huggingface_repo_id is not None:

  cfg.output.logging (2 occurrences)
    L 254: setup_logging(cfg.output.logging, reset=True)  # pyright: ignore[reportAttributeAccessIssue]
    L 269: cfg.performance, cfg.output.logging, training_config

  cfg.output.logging.log_tracker_config (2 occurrences)
    L 599: if cfg.output.logging.log_tracker_config is not None:
    L 600: init_kwargs = cfg.output.logging.log_tracker_config

  cfg.output.logging.log_tracker_name (2 occurrences)
    L 603: if cfg.output.logging.log_tracker_name is None
    L 604: else cfg.output.logging.log_tracker_name,

  cfg.output.logging.wandb_run_name (2 occurrences)
    L 597: if cfg.output.logging.wandb_run_name:
    L 598: init_kwargs["wandb"] = {"name": cfg.output.logging.wandb_run_name}

  cfg.output.metadata (1 occurrences)
    L 617: metadata_config=cfg.output.metadata,

  cfg.output.sampling (3 occurrences)
    L 642: cfg.output.sampling,
    L 819: cfg.output.sampling,
    L 943: cfg.output.sampling,

  cfg.output.saving (4 occurrences)
    L 246: saving_config = cfg.output.saving
    L 644: cfg.output.saving,
    L 821: cfg.output.saving,
    L 945: cfg.output.saving,

### PERFORMANCE ###
----------------------------------------

  cfg.performance (3 occurrences)
    L 269: cfg.performance, cfg.output.logging, training_config
    L 272: weight_dtype, save_dtype = prepare_dtype(cfg.performance, saving_config)
    L 278: model_config, cfg.performance, weight_dtype, accelerator

  cfg.performance.attention.mem_eff_attn (1 occurrences)
    L 416: cfg.performance.attention.mem_eff_attn,

  cfg.performance.attention.sdpa (1 occurrences)
    L 418: cfg.performance.attention.sdpa,

  cfg.performance.attention.xformers (2 occurrences)
    L 417: cfg.performance.attention.xformers,
    L 421: vae.set_use_memory_efficient_attention_xformers(cfg.performance.attention.xformers)

  cfg.performance.precision.no_half_vae (1 occurrences)
    L 274: torch.float32 if cfg.performance.precision.no_half_vae else weight_dtype

### TEXTUAL_INVERSION ###
----------------------------------------

  cfg.textual_inversion (1 occurrences)
    L 241: ti_config = cfg.textual_inversion

### TIMESTEP ###
----------------------------------------

  cfg.timestep (1 occurrences)
    L 704: cfg.timestep,

### TRAINING ###
----------------------------------------

  cfg.training (5 occurrences)
    L 242: training_config = cfg.training
    L 474: cfg.training,
    L 643: cfg.training,
    L 820: cfg.training,
    L 944: cfg.training,

================================================================================
Total unique patterns: 30
================================================================================