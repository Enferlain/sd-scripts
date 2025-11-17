import os
import argparse
import logging
import torch

from diffusers import StableDiffusionPipeline

from library.models import model_util
from library.models.original_unet import UNet2DConditionModel
from library.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)


# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE


# def replace_unet_modules(unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, mem_eff_attn, xformers):
#     replace_attentions_for_hypernetwork()
#     # unet is not used currently, but it is here for future use
#     unet.enable_xformers_memory_efficient_attention()
#     return
#     if mem_eff_attn:
#         unet.set_attn_processor(FlashAttnProcessor())
#     elif xformers:
#         unet.enable_xformers_memory_efficient_attention()


# def replace_unet_cross_attn_to_xformers():
#     logger.info("CrossAttention.forward has been replaced to enable xformers.")
#     try:
#         import xformers.ops
#     except ImportError:
#         raise ImportError("No xformers / xformersがインストールされていないようです")

#     def forward_xformers(self, x, context=None, mask=None):
#         h = self.heads
#         q_in = self.to_q(x)

#         context = default(context, x)
#         context = context.to(x.dtype)

#         if hasattr(self, "hypernetwork") and self.hypernetwork is not None:
#             context_k, context_v = self.hypernetwork.forward(x, context)
#             context_k = context_k.to(x.dtype)
#             context_v = context_v.to(x.dtype)
#         else:
#             context_k = context
#             context_v = context

#         k_in = self.to_k(context_k)
#         v_in = self.to_v(context_v)

#         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
#         del q_in, k_in, v_in

#         q = q.contiguous()
#         k = k.contiguous()
#         v = v.contiguous()
#         out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)  # 最適なのを選んでくれる

#         out = rearrange(out, "b n h d -> b n (h d)", h=h)

#         # diffusers 0.7.0~
#         out = self.to_out[0](out)
#         out = self.to_out[1](out)
#         return out


#  diffusers.models.attention.CrossAttention.forward = forward_xformers
def replace_unet_modules(unet: UNet2DConditionModel, mem_eff_attn, xformers, sdpa):
    if mem_eff_attn:
        logger.info("Enable memory efficient attention for U-Net")
        unet.set_use_memory_efficient_attention(False, True)
    elif xformers:
        logger.info("Enable xformers for U-Net")
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("No xformers / xformersがインストールされていないようです")

        unet.set_use_memory_efficient_attention(True, False)
    elif sdpa:
        logger.info("Enable SDPA for U-Net")
        unet.set_use_sdpa(True)


"""
def replace_vae_modules(vae: diffusers.models.AutoencoderKL, mem_eff_attn, xformers):
    # vae is not used currently, but it is here for future use
    if mem_eff_attn:
        replace_vae_attn_to_memory_efficient()
    elif xformers:
        # とりあえずDiffusersのxformersを使う。AttentionがあるのはMidBlockのみ
        logger.info("Use Diffusers xformers for VAE")
        vae.encoder.mid_block.attentions[0].set_use_memory_efficient_attention_xformers(True)
        vae.decoder.mid_block.attentions[0].set_use_memory_efficient_attention_xformers(True)


def replace_vae_attn_to_memory_efficient():
    logger.info("AttentionBlock.forward has been replaced to FlashAttention (not xformers)")
    flash_func = FlashAttentionFunction

    def forward_flash_attn(self, hidden_states):
        logger.info("forward_flash_attn")
        q_bucket_size = 512
        k_bucket_size = 1024

        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        query_proj, key_proj, value_proj = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (query_proj, key_proj, value_proj)
        )

        out = flash_func.apply(query_proj, key_proj, value_proj, None, False, q_bucket_size, k_bucket_size)

        out = rearrange(out, "b h n d -> b n (h d)")

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    diffusers.models.attention.AttentionBlock.forward = forward_flash_attn
"""


def set_padding_mode_for_vae_conv2d_modules(vae: torch.nn.Module, padding_mode: str = 'zeros'):
    """Apply padding mode only to Conv2d modules with non-zero padding (for EQ VAE)"""
    logger.info(f"VAE padding mode set to: {padding_mode}")
    for name, module in vae.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pad = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
            if pad[0] > 0 or pad[1] > 0:
                # print(f"Applying padding mode '{padding_mode}' to {name} ({module.__class__.__name__})")
                module.padding_mode = padding_mode


def _load_target_model(args: argparse.Namespace, weight_dtype, device="cpu", unet_use_linear_projection_in_v2=False):
    name_or_path = args.pretrained_model_name_or_path
    name_or_path = os.path.realpath(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers
    if load_stable_diffusion_format:
        logger.info(f"load StableDiffusion checkpoint: {name_or_path}")
        text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(
            args.v2, name_or_path, device, unet_use_linear_projection_in_v2=unet_use_linear_projection_in_v2
        )
    else:
        # Diffusers model is loaded to CPU
        logger.info(f"load Diffusers pretrained models: {name_or_path}")
        try:
            pipe = StableDiffusionPipeline.from_pretrained(name_or_path, tokenizer=None, safety_checker=None)
        except EnvironmentError as ex:
            logger.error(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}"
            )
            raise ex
        text_encoder = pipe.text_encoder
        vae = pipe.vae
        unet = pipe.unet
        del pipe

        # Diffusers U-Net to original U-Net
        # TODO *.ckpt/*.safetensorsのv2と同じ形式にここで変換すると良さそう
        # logger.info(f"unet config: {unet.config}")
        original_unet = UNet2DConditionModel(
            unet.config.sample_size,
            unet.config.attention_head_dim,
            unet.config.cross_attention_dim,
            unet.config.use_linear_projection,
            unet.config.upcast_attention,
        )
        original_unet.load_state_dict(unet.state_dict())
        unet = original_unet
        logger.info("U-Net converted to original U-Net")

    # VAEを読み込む
    if args.vae is not None:
        vae = model_util.load_vae(args.vae, weight_dtype)
        logger.info("additional VAE loaded")

    if hasattr(args, "vae_conv2d_padding_mode") and args.vae_conv2d_padding_mode is not None and args.vae_conv2d_padding_mode.lower() != 'zeros':
        logger.info(f"Loaded VAE with padding mode: {args.vae_conv2d_padding_mode}")
        set_padding_mode_for_vae_conv2d_modules(vae, args.vae_conv2d_padding_mode)

    return text_encoder, vae, unet, load_stable_diffusion_format


def load_target_model(args, weight_dtype, accelerator, unet_use_linear_projection_in_v2=False):
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.info(
                f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            text_encoder, vae, unet, load_stable_diffusion_format = _load_target_model(
                args,
                weight_dtype,
                accelerator.device if args.lowram else "cpu",
                unet_use_linear_projection_in_v2=unet_use_linear_projection_in_v2,
            )
            # work on low-ram device
            if args.lowram:
                text_encoder.to(accelerator.device)
                unet.to(accelerator.device)
                vae.to(accelerator.device)

            clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()
    return text_encoder, vae, unet, load_stable_diffusion_format


def patch_accelerator_for_fp16_training(accelerator):
    from accelerate import DistributedType

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        return

    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer
