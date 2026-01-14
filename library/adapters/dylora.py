# some codes are copied from:
# https://github.com/huawei-noah/KD-NLP/blob/main/DyLoRA/

# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Changes made to the original code:
# 2022.08.20 - Integrate the DyLoRA layer for the LoRA Linear layer
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import math
import os
import random
import torch

from diffusers import AutoencoderKL
from transformers import CLIPTextModel
from torch import nn

from library.config.dataclasses.optimizer import LearningRatesConfig
from library.utils.hash_utils import precalculate_safetensors_hashes
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class DyLoRAModule(torch.nn.Module):
    """
    DyLoRA module that replaces the forward method of the original Linear or Conv2d module.
    It implements Dynamic Low-Rank Adaptation (DyLoRA).
    """

    # NOTE: support dropout in future
    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_dim=4, alpha=1, unit=1):
        """
        Initialize the DyLoRAModule.

        Args:
            lora_name (str): The name of the LoRA module.
            org_module (torch.nn.Module): The original module to be adapted.
            multiplier (float, optional): The multiplier for the LoRA output. Defaults to 1.0.
            lora_dim (int, optional): The dimension (rank) of the LoRA. Defaults to 4.
            alpha (float, optional): The alpha parameter for LoRA scaling. Defaults to 1.
            unit (int, optional): The unit for dynamic rank selection. Defaults to 1.
        """
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.unit = unit
        assert self.lora_dim % self.unit == 0, "rank must be a multiple of unit"

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # can be treated as a constant

        self.is_conv2d = org_module.__class__.__name__ == "Conv2d"
        self.is_conv2d_3x3 = self.is_conv2d and org_module.kernel_size == (3, 3)

        if self.is_conv2d and self.is_conv2d_3x3:
            kernel_size = org_module.kernel_size
            self.stride = org_module.stride
            self.padding = org_module.padding
            self.lora_A = nn.ParameterList([org_module.weight.new_zeros((1, in_dim, *kernel_size)) for _ in range(self.lora_dim)])
            self.lora_B = nn.ParameterList([org_module.weight.new_zeros((out_dim, 1, 1, 1)) for _ in range(self.lora_dim)])
        else:
            self.lora_A = nn.ParameterList([org_module.weight.new_zeros((1, in_dim)) for _ in range(self.lora_dim)])
            self.lora_B = nn.ParameterList([org_module.weight.new_zeros((out_dim, 1)) for _ in range(self.lora_dim)])

        # same as microsoft's
        for lora in self.lora_A:
            torch.nn.init.kaiming_uniform_(lora, a=math.sqrt(5))
        for lora in self.lora_B:
            torch.nn.init.zeros_(lora)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        """
        Apply the DyLoRA module to the original module by replacing its forward method.
        """
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        """
        Forward pass of the DyLoRA module.
        Randomly selects a rank for training.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with LoRA adaptation applied.
        """
        result = self.org_forward(x)

        # specify the dynamic rank
        trainable_rank = random.randint(0, self.lora_dim - 1)
        trainable_rank = trainable_rank - trainable_rank % self.unit  # make sure the rank is a multiple of unit

        # freeze some parameters and train the rest
        for i in range(0, trainable_rank):
            self.lora_A[i].requires_grad = False
            self.lora_B[i].requires_grad = False
        for i in range(trainable_rank, trainable_rank + self.unit):
            self.lora_A[i].requires_grad = True
            self.lora_B[i].requires_grad = True
        for i in range(trainable_rank + self.unit, self.lora_dim):
            self.lora_A[i].requires_grad = False
            self.lora_B[i].requires_grad = False

        lora_A = torch.cat(tuple(self.lora_A), dim=0)
        lora_B = torch.cat(tuple(self.lora_B), dim=1)

        # calculate with lora_A and lora_B
        if self.is_conv2d_3x3:
            ab = torch.nn.functional.conv2d(x, lora_A, stride=self.stride, padding=self.padding)
            ab = torch.nn.functional.conv2d(ab, lora_B)
        else:
            ab = x
            if self.is_conv2d:
                ab = ab.reshape(ab.size(0), ab.size(1), -1).transpose(1, 2)  # (N, C, H, W) -> (N, H*W, C)

            ab = torch.nn.functional.linear(ab, lora_A)
            ab = torch.nn.functional.linear(ab, lora_B)

            if self.is_conv2d:
                ab = ab.transpose(1, 2).reshape(ab.size(0), -1, *x.size()[2:])  # (N, H*W, C) -> (N, C, H, W)

        # last term is scaling to make low rank larger (probably)
        result = result + ab * self.scale * math.sqrt(self.lora_dim / (trainable_rank + self.unit))

        # NOTE might be faster to add to weight before calling linear/conv2d
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Returns a dictionary containing a whole state of the module.
        The state dict is formatted to be compatible with standard LoRA state dicts.

        Args:
            destination (dict, optional): If provided, the state of module will be updated into the dict and the same object is returned. Otherwise, an OrderedDict will be created and returned.
            prefix (str, optional): a prefix string that will be added to the key of the state.
            keep_vars (bool, optional): by default the Tensor s returned in the state dict are detached from the parameter history.

        Returns:
            dict: The state dictionary.
        """
        # make state dict same as normal LoRA:
        # nn.ParameterList becomes like .lora_A.0, so cat and replace like in forward
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        lora_A_weight = torch.cat(tuple(self.lora_A), dim=0)
        if self.is_conv2d and not self.is_conv2d_3x3:
            lora_A_weight = lora_A_weight.unsqueeze(-1).unsqueeze(-1)

        lora_B_weight = torch.cat(tuple(self.lora_B), dim=1)
        if self.is_conv2d and not self.is_conv2d_3x3:
            lora_B_weight = lora_B_weight.unsqueeze(-1).unsqueeze(-1)

        sd[self.lora_name + ".lora_down.weight"] = lora_A_weight if keep_vars else lora_A_weight.detach()
        sd[self.lora_name + ".lora_up.weight"] = lora_B_weight if keep_vars else lora_B_weight.detach()

        i = 0
        while True:
            key_a = f"{self.lora_name}.lora_A.{i}"
            key_b = f"{self.lora_name}.lora_B.{i}"
            if key_a in sd:
                sd.pop(key_a)
                sd.pop(key_b)
            else:
                break
            i += 1
        return sd

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Loads the module state from a state dictionary.
        Compatible with standard LoRA state dicts.
        """
        # make it possible to load the same state dict as normal LoRA: asked chatGPT for this method
        lora_A_weight = state_dict.pop(self.lora_name + ".lora_down.weight", None)
        lora_B_weight = state_dict.pop(self.lora_name + ".lora_up.weight", None)

        if lora_A_weight is None or lora_B_weight is None:
            if strict:
                raise KeyError(f"{self.lora_name}.lora_down/up.weight is not found")
            else:
                return

        if self.is_conv2d and not self.is_conv2d_3x3:
            lora_A_weight = lora_A_weight.squeeze(-1).squeeze(-1)
            lora_B_weight = lora_B_weight.squeeze(-1).squeeze(-1)

        state_dict.update(
            {f"{self.lora_name}.lora_A.{i}": nn.Parameter(lora_A_weight[i].unsqueeze(0)) for i in range(lora_A_weight.size(0))}
        )
        state_dict.update(
            {f"{self.lora_name}.lora_B.{i}": nn.Parameter(lora_B_weight[:, i].unsqueeze(1)) for i in range(lora_B_weight.size(1))}
        )

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def create_adapter(
    multiplier: float,
    adapter_rank: int | None,
    adapter_alpha: float | None,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel | list[CLIPTextModel],
    unet,
    **kwargs,
):
    """
    Creates a DyLoRA adapter.

    Args:
        multiplier (float): Multiplier for the adapter output.
        adapter_rank (int, optional): Rank of the adapter (lora_dim). Defaults to 4.
        adapter_alpha (float, optional): Alpha parameter for scaling. Defaults to 1.0.
        vae (AutoencoderKL): VAE model (not used in DyLoRA creation but kept for interface consistency).
        text_encoder (Union[CLIPTextModel, List[CLIPTextModel]]): Text encoder(s).
        unet (UNet2DConditionModel): U-Net model.
        **kwargs: Additional arguments, including 'conv_dim', 'conv_alpha', 'unit', and LoRA+ ratios.

    Returns:
        DyLoRAAdapter: The created DyLoRA adapter.
    """
    if adapter_rank is None:
        adapter_rank = 4  # default
    if adapter_alpha is None:
        adapter_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim")
    conv_alpha = kwargs.get("conv_alpha")
    unit = kwargs.get("unit")
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        assert conv_dim == adapter_rank, "conv_dim must be same as dim"
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    if unit is not None:
        unit = int(unit)
    else:
        unit = 1

    adapter = DyLoRAAdapter(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=adapter_rank,
        alpha=adapter_alpha,
        apply_to_conv=conv_dim is not None,
        unit=unit,
        varbose=True,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio")
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio")
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio")
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        adapter.set_loraplus_lr_ratio(loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    return adapter


# Create peft from weights for inference, weights are not loaded here (because can be merged)
def create_adapter_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    """
    Creates a DyLoRA adapter from weights for inference.

    Args:
        multiplier (float): Multiplier for the adapter output.
        file (str): Path to the weights file.
        vae (AutoencoderKL): VAE model.
        text_encoder (Union[CLIPTextModel, List[CLIPTextModel]]): Text encoder(s).
        unet (UNet2DConditionModel): U-Net model.
        weights_sd (dict, optional): State dict of weights. If None, loaded from file.
        for_inference (bool, optional): Whether to create for inference. Defaults to False.
        **kwargs: Additional arguments.

    Returns:
        tuple: (DyLoRAAdapter, dict) The created adapter and the weights state dict.
    """
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
            # logger.info(f"{lora_name} {value.size()} {dim}")

    # support old LoRA without alpha
    for key in modules_dim:
        if key not in modules_alpha:
            modules_alpha = modules_dim[key]

    module_class = DyLoRAModule

    adapter = DyLoRAAdapter(
        text_encoder, unet, multiplier=multiplier, modules_dim=modules_dim, modules_alpha=modules_alpha, module_class=module_class
    )
    return adapter, weights_sd


class DyLoRAAdapter(torch.nn.Module):
    """
    Adapter class for DyLoRA (Dynamic Low-Rank Adaptation).
    Manages the application and training of DyLoRA modules on Text Encoder and U-Net.
    """

    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(
        self,
        text_encoder,
        unet,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        apply_to_conv=False,
        modules_dim=None,
        modules_alpha=None,
        unit=1,
        module_class=DyLoRAModule,
        varbose=False,
    ) -> None:
        """
        Initialize the DyLoRAAdapter.

        Args:
            text_encoder (Union[CLIPTextModel, List[CLIPTextModel]]): The text encoder model(s).
            unet (UNet2DConditionModel): The U-Net model.
            multiplier (float, optional): The multiplier for the adapter. Defaults to 1.0.
            lora_dim (int, optional): The dimension (rank) of the LoRA. Defaults to 4.
            alpha (float, optional): The alpha parameter for LoRA scaling. Defaults to 1.
            apply_to_conv (bool, optional): Whether to apply LoRA to Conv2d layers. Defaults to False.
            modules_dim (dict, optional): Dictionary mapping module names to dimensions (ranks) for loading from weights.
            modules_alpha (dict, optional): Dictionary mapping module names to alpha values for loading from weights.
            unit (int, optional): The unit for dynamic rank selection. Defaults to 1.
            module_class (type, optional): The class to use for LoRA modules. Defaults to DyLoRAModule.
            varbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.apply_to_conv = apply_to_conv

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info("create LoRA peft from weights")
        else:
            logger.info(f"create LoRA peft. base dim (rank): {lora_dim}, alpha: {alpha}, unit: {unit}")
            if self.apply_to_conv:
                logger.info("apply LoRA to Conv2d with kernel size (3,3).")

        # create module instances
        def create_modules(is_unet, root_module: torch.nn.Module, target_replace_modules) -> list[DyLoRAModule]:
            prefix = DyLoRAAdapter.LORA_PREFIX_UNET if is_unet else DyLoRAAdapter.LORA_PREFIX_TEXT_ENCODER
            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            dim = None
                            alpha = None
                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            else:
                                if is_linear or is_conv2d_1x1 or apply_to_conv:
                                    dim = self.lora_dim
                                    alpha = self.alpha

                            if dim is None or dim == 0:
                                continue

                            # dropout and fan_in_fan_out is default
                            lora = module_class(lora_name, child_module, self.multiplier, dim, alpha, unit)
                            loras.append(lora)
            return loras

        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        self.text_encoder_loras = []
        for i, text_encoder in enumerate(text_encoders):
            if len(text_encoders) > 1:
                index = i + 1
                logger.info(f"create LoRA for Text Encoder {index}")
            else:
                index = None
                logger.info("create LoRA for Text Encoder")

            text_encoder_loras = create_modules(False, text_encoder, DyLoRAAdapter.TEXT_ENCODER_TARGET_REPLACE_MODULE)
            self.text_encoder_loras.extend(text_encoder_loras)

        # self.text_encoder_loras = create_modules(False, text_encoder, DyLoRAAdapter.TEXT_ENCODER_TARGET_REPLACE_MODULE)
        logger.info(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
        target_modules = DyLoRAAdapter.UNET_TARGET_REPLACE_MODULE
        if modules_dim is not None or self.apply_to_conv:
            target_modules += DyLoRAAdapter.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        self.unet_loras = create_modules(True, unet, target_modules)
        logger.info(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        """
        Sets the learning rate ratios for LoRA+.

        Args:
            loraplus_lr_ratio (float): General LoRA+ learning rate ratio.
            loraplus_unet_lr_ratio (float): LoRA+ learning rate ratio specifically for U-Net.
            loraplus_text_encoder_lr_ratio (float): LoRA+ learning rate ratio specifically for Text Encoder.
        """
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}")
        logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def set_multiplier(self, multiplier):
        """
        Sets the multiplier for all LoRA modules.

        Args:
            multiplier (float): The new multiplier value.
        """
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def load_weights(self, file):
        """
        Loads weights from a file (safetensors or torch).

        Args:
            file (str): Path to the weights file.

        Returns:
            The result of load_state_dict.
        """
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        """
        Applies the LoRA adapter to the Text Encoder and U-Net.

        Args:
            text_encoder: Text Encoder model (unused but kept for signature compatibility).
            unet: U-Net model (unused but kept for signature compatibility).
            apply_text_encoder (bool, optional): Whether to apply to Text Encoder. Defaults to True.
            apply_unet (bool, optional): Whether to apply to U-Net. Defaults to True.
        """
        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    """
    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(DyLoRAAdapter.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(DyLoRAAdapter.LORA_PREFIX_UNET):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)

        logger.info(f"weights are merged")
    """

    # might be good to allow setting different learning rates for two Text Encoders
    def prepare_optimizer_params(self, learning_rates: LearningRatesConfig, apply_orthograd: bool, orthograd_targets: list[str]):
        """
        Prepares optimizer parameters for training.

        Args:
            learning_rates (LearningRatesConfig): Configuration for learning rates.
            apply_orthograd (bool): Whether to apply OrthoGrad.
            orthograd_targets (list[str]): List of targets for OrthoGrad.

        Returns:
            list: List of parameter groups for the optimizer.
        """
        # Extract LRs from config
        unet_lr = learning_rates.unet
        base_lr = learning_rates.base
        # Handle text_encoders which may be float, list, or None
        raw_te_lr = learning_rates.text_encoders
        if raw_te_lr is None or isinstance(raw_te_lr, (float, int)):
            text_encoder_lr = raw_te_lr
        else:
            # List - take first element for single-TE adapters
            text_encoder_lr = raw_te_lr[0] if len(raw_te_lr) > 0 else None

        self.requires_grad_(True)
        all_params = []

        def assemble_params(loras, lr, ratio):
            param_groups = {"lora": {}, "plus": {}}
            for lora in loras:
                for name, param in lora.named_parameters():
                    if ratio is not None and "lora_B" in name:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            for key in param_groups:
                param_data = {"params": param_groups[key].values()}

                if len(param_data["params"]) == 0:
                    continue

                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * ratio
                    else:
                        param_data["lr"] = lr

                if param_data.get("lr") == 0 or param_data.get("lr") is None:
                    continue

                params.append(param_data)

            return params

        if self.text_encoder_loras:
            params = assemble_params(
                self.text_encoder_loras,
                text_encoder_lr if text_encoder_lr is not None else base_lr,
                self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)

        if self.unet_loras:
            params = assemble_params(
                self.unet_loras, base_lr if unet_lr is None else unet_lr, self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio
            )
            all_params.extend(params)

        return all_params

    def enable_gradient_checkpointing(self):
        """
        Enables gradient checkpointing (not supported for DyLoRA).
        """
        # not supported
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        """
        Prepares for gradient calculation. Sets requires_grad to True.
        """
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        """
        Called at the start of each epoch. Sets the model to train mode.
        """
        self.train()

    def get_trainable_params(self):
        """
        Returns the trainable parameters of the adapter.
        """
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        """
        Saves the adapter weights to a file.

        Args:
            file (str): Path to the output file.
            dtype (torch.dtype): Data type to save weights in.
            metadata (dict): Metadata to save with the weights (for safetensors).
        """
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    # mask is a tensor with values from 0 to 1
    def set_region(self, sub_prompt_index, is_last_adapter, mask):
        """
        Sets the region for regional LoRA (not implemented for DyLoRA).
        """
        pass

    def set_current_generation(self, batch_size, num_sub_prompts, width, height, shared):
        """
        Sets the current generation parameters (not implemented for DyLoRA).
        """
        pass
