from dataclasses import dataclass, field


@dataclass
class TextualInversionConfig:
    """Textual inversion training specific configuration."""

    weights: str | None = field(default=None, metadata={"help": "Path to existing embeddings file to continue training from"})
    num_vectors_per_token: int = field(
        default=1, metadata={"help": "Number of vectors per token (1 for simple, higher for complex concepts)"}
    )
    token_string: str | None = field(default=None, metadata={"help": "Trigger word/token for the trained embedding"})
    init_word: str | None = field(default=None, metadata={"help": "Initialize embedding from this word's vectors"})
    use_object_template: bool = field(default=False, metadata={"help": "Use object-style caption templates for training"})
    use_style_template: bool = field(default=False, metadata={"help": "Use style-focused caption templates for training"})
