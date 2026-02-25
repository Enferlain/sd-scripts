from dataclasses import dataclass, field


@dataclass
class ValidationConfig:
    """
    Unified validation configuration.
    Contains both data splitting settings and validation loop settings.
    """

    # Data splitting
    validation_split: float = field(default=0.0, metadata={"help": "Split for validation images out of the training dataset"})
    validation_seed: int | None = field(
        default=None, metadata={"help": "Validation seed for shuffling validation dataset, training seed used otherwise"}
    )

    # Validation scheduling
    run_at_start: bool = field(default=False, metadata={"help": "Run validation at training start (step 0)"})
    run_at_end: bool = field(default=False, metadata={"help": "Run validation at training end"})
    validate_every_n_steps: int | None = field(default=None, metadata={"help": "Run validation on validation dataset every N steps"})
    validate_every_n_epochs: int | None = field(default=None, metadata={"help": "Run validation on validation dataset every N epochs"})

    # Validation loop execution
    max_validation_steps: int | None = field(default=None, metadata={"help": "Max number of validation dataset items processed"})
    validation_timesteps: str = field(
        default="[50, 350, 500, 650, 950]", metadata={"help": "A list of timesteps to use for each validation step"}
    )
