from torchao.optim.adam import AdamW4bit, AdamW8bit, AdamWFp8


class AdamW8bitAO(AdamW8bit):
    """TorchAO AdamW8bit exposed through the repo's optimizer registry."""

    # Just a wrapper to rename the upstream TorchAO class into the repo-facing
    # optimizer family naming used by existing configs and vendor references.
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=256,
        bf16_stochastic_round=False,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
        )

    def __str__(self) -> str:
        return "AdamW8bitAO"


class AdamW4bitAO(AdamW4bit):
    """TorchAO AdamW4bit exposed through the repo's optimizer registry."""

    # Keep the vendor-facing defaults intact so the absorbed family behaves the
    # same way as the donor implementation unless we intentionally change it.
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=128,
        bf16_stochastic_round=False,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
        )

    def __str__(self) -> str:
        return "AdamW4bitAO"


class AdamWfp8AO(AdamWFp8):
    """TorchAO AdamWFp8 exposed through the repo's optimizer registry."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=256,
        bf16_stochastic_round=False,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
        )

    def __str__(self) -> str:
        return "AdamWfp8AO"
