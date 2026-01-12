import logging
import diffusers
import torch

from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from safetensors.torch import load_file, save_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from library.utils.common_utils import setup_logging
from library.models.sd.unet import UNet2DConditionModel
from library.constants import (
    DIFFUSERS_REF_MODEL_ID_V2,
    DIFFUSERS_REF_MODEL_ID_V1,
    UNET_PARAMS_MODEL_CHANNELS,
    UNET_PARAMS_CHANNEL_MULT,
    UNET_PARAMS_ATTENTION_RESOLUTIONS,
    UNET_PARAMS_IMAGE_SIZE,
    UNET_PARAMS_IN_CHANNELS,
    UNET_PARAMS_OUT_CHANNELS,
    UNET_PARAMS_NUM_RES_BLOCKS,
    UNET_PARAMS_CONTEXT_DIM,
    V2_UNET_PARAMS_CONTEXT_DIM,
    UNET_PARAMS_NUM_HEADS,
    V2_UNET_PARAMS_ATTENTION_HEAD_DIM,
)

from library.models.sd.vae import (
    create_vae_diffusers_config,
    convert_ldm_vae_checkpoint,
    convert_vae_state_dict,
    assign_to_checkpoint,
)
from library.models.common import shave_segments, renew_attention_paths, renew_resnet_paths
from library.utils.safetensors_utils import is_safetensors

setup_logging()
logger = logging.getLogger(__name__)


def convert_ldm_clip_checkpoint_v1(checkpoint):
    """
    Converts a V1 LDM CLIP checkpoint to the Diffusers format.

    Args:
        checkpoint (dict): The source checkpoint state dictionary.

    Returns:
        dict: The converted state dictionary for the text model.
    """
    keys = list(checkpoint.keys())
    text_model_dict = {}
    for key in keys:
        if key.startswith("cond_stage_model.transformer"):
            text_model_dict[key[len("cond_stage_model.transformer.") :]] = checkpoint[key]

    # remove position_ids for newer transformer, which causes error :(
    if "text_model.embeddings.position_ids" in text_model_dict:
        text_model_dict.pop("text_model.embeddings.position_ids")

    return text_model_dict


def convert_ldm_clip_checkpoint_v2(checkpoint, max_length):
    """
    Converts a V2 LDM CLIP checkpoint to the Diffusers format.

    Args:
        checkpoint (dict): The source checkpoint state dictionary.
        max_length (int): Maximum length of the sequence (unused in function body but kept for signature).

    Returns:
        dict: The converted state dictionary for the text model.
    """

    # 嫌になるくらい違うぞ！
    def convert_key(key):
        if not key.startswith("cond_stage_model"):
            return None

        # common conversion
        key = key.replace("cond_stage_model.model.transformer.", "text_model.encoder.")
        key = key.replace("cond_stage_model.model.", "text_model.")

        if "resblocks" in key:
            # resblocks conversion
            key = key.replace(".resblocks.", ".layers.")
            if ".ln_" in key:
                key = key.replace(".ln_", ".layer_norm")
            elif ".mlp." in key:
                key = key.replace(".c_fc.", ".fc1.")
                key = key.replace(".c_proj.", ".fc2.")
            elif ".attn.out_proj" in key:
                key = key.replace(".attn.out_proj.", ".self_attn.out_proj.")
            elif ".attn.in_proj" in key:
                key = None  # 特殊なので後で処理する
            else:
                raise ValueError(f"unexpected key in SD: {key}")
        elif ".positional_embedding" in key:
            key = key.replace(".positional_embedding", ".embeddings.position_embedding.weight")
        elif ".text_projection" in key or ".logit_scale" in key:
            key = None  # 使われない???
        elif ".token_embedding" in key:
            key = key.replace(".token_embedding.weight", ".embeddings.token_embedding.weight")
        elif ".ln_final" in key:
            key = key.replace(".ln_final", ".final_layer_norm")
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        # remove resblocks 23
        if ".resblocks.23." in key:
            continue
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if ".resblocks.23." in key:
            continue
        if ".resblocks" in key and ".attn.in_proj_" in key:
            # 三つに分割
            values = torch.chunk(checkpoint[key], 3)

            key_suffix = ".weight" if "weight" in key else ".bias"
            key_pfx = key.replace("cond_stage_model.model.transformer.resblocks.", "text_model.encoder.layers.")
            key_pfx = key_pfx.replace("_weight", "")
            key_pfx = key_pfx.replace("_bias", "")
            key_pfx = key_pfx.replace(".attn.in_proj", ".self_attn.")
            new_sd[key_pfx + "q_proj" + key_suffix] = values[0]
            new_sd[key_pfx + "k_proj" + key_suffix] = values[1]
            new_sd[key_pfx + "v_proj" + key_suffix] = values[2]

    # remove position_ids for newer transformer, which causes error :(
    ANOTHER_POSITION_IDS_KEY = "text_model.encoder.text_model.embeddings.position_ids"
    if ANOTHER_POSITION_IDS_KEY in new_sd:
        # waifu diffusion v1.4
        del new_sd[ANOTHER_POSITION_IDS_KEY]

    if "text_model.embeddings.position_ids" in new_sd:
        del new_sd["text_model.embeddings.position_ids"]

    return new_sd


def conv_transformer_to_linear(checkpoint):
    """
    Converts 1x1 convolutional weights in transformers to linear weights.

    Args:
        checkpoint (dict): The state dictionary to modify in-place.
    """
    keys = list(checkpoint.keys())
    tf_keys = ["proj_in.weight", "proj_out.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in tf_keys and checkpoint[key].ndim > 2:
            checkpoint[key] = checkpoint[key][:, :, 0, 0]


def convert_unet_state_dict_to_sd(v2, unet_state_dict):
    """
    Converts a Diffusers UNet state dictionary to a Stable Diffusion LDM format.

    Args:
        v2 (bool): Whether the model is SD v2.
        unet_state_dict (dict): The Diffusers UNet state dictionary.

    Returns:
        dict: The converted Stable Diffusion LDM state dictionary.
    """
    unet_conversion_map = [
        # (stable-diffusion, HF Diffusers)
        ("time_embed.0.weight", "time_embedding.linear_1.weight"),
        ("time_embed.0.bias", "time_embedding.linear_1.bias"),
        ("time_embed.2.weight", "time_embedding.linear_2.weight"),
        ("time_embed.2.bias", "time_embedding.linear_2.bias"),
        ("input_blocks.0.0.weight", "conv_in.weight"),
        ("input_blocks.0.0.bias", "conv_in.bias"),
        ("out.0.weight", "conv_norm_out.weight"),
        ("out.0.bias", "conv_norm_out.bias"),
        ("out.2.weight", "conv_out.weight"),
        ("out.2.bias", "conv_out.bias"),
    ]

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0", "norm1"),
        ("in_layers.2", "conv1"),
        ("out_layers.0", "norm2"),
        ("out_layers.3", "conv2"),
        ("emb_layers.1", "time_emb_proj"),
        ("skip_connection", "conv_shortcut"),
    ]

    unet_conversion_map_layer = []
    for i in range(4):
        # loop over downblocks/upblocks

        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3 * i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            if i > 0:
                # no attention layers in up_blocks.0
                hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
                sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
                unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3 * (i + 1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3 * i + 2}.{1 if i == 0 else 2}."
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2 * j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}

    if v2:
        conv_transformer_to_linear(new_state_dict)

    return new_state_dict


def controlnet_conversion_map():
    """
    Creates a mapping for converting ControlNet models.

    Returns:
        tuple: A tuple containing (unet_conversion_map, unet_conversion_map_resnet, unet_conversion_map_layer).
    """
    unet_conversion_map = [
        ("time_embed.0.weight", "time_embedding.linear_1.weight"),
        ("time_embed.0.bias", "time_embedding.linear_1.bias"),
        ("time_embed.2.weight", "time_embedding.linear_2.weight"),
        ("time_embed.2.bias", "time_embedding.linear_2.bias"),
        ("input_blocks.0.0.weight", "conv_in.weight"),
        ("input_blocks.0.0.bias", "conv_in.bias"),
        ("middle_block_out.0.weight", "controlnet_mid_block.weight"),
        ("middle_block_out.0.bias", "controlnet_mid_block.bias"),
    ]

    unet_conversion_map_resnet = [
        ("in_layers.0", "norm1"),
        ("in_layers.2", "conv1"),
        ("out_layers.0", "norm2"),
        ("out_layers.3", "conv2"),
        ("emb_layers.1", "time_emb_proj"),
        ("skip_connection", "conv_shortcut"),
    ]

    unet_conversion_map_layer = []
    for i in range(4):
        for j in range(2):
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3 * i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        if i < 3:
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3 * (i + 1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2 * j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    controlnet_cond_embedding_names = ["conv_in"] + [f"blocks.{i}" for i in range(6)] + ["conv_out"]
    for i, hf_prefix in enumerate(controlnet_cond_embedding_names):
        hf_prefix = f"controlnet_cond_embedding.{hf_prefix}."
        sd_prefix = f"input_hint_block.{i * 2}."
        unet_conversion_map_layer.append((sd_prefix, hf_prefix))

    for i in range(12):
        hf_prefix = f"controlnet_down_blocks.{i}."
        sd_prefix = f"zero_convs.{i}.0."
        unet_conversion_map_layer.append((sd_prefix, hf_prefix))

    return unet_conversion_map, unet_conversion_map_resnet, unet_conversion_map_layer


def convert_controlnet_state_dict_to_sd(controlnet_state_dict):
    """
    Converts a Diffusers ControlNet state dictionary to a Stable Diffusion LDM format.

    Args:
        controlnet_state_dict (dict): The Diffusers ControlNet state dictionary.

    Returns:
        dict: The converted Stable Diffusion LDM state dictionary.
    """
    unet_conversion_map, unet_conversion_map_resnet, unet_conversion_map_layer = controlnet_conversion_map()

    mapping = {k: k for k in controlnet_state_dict}
    for sd_name, diffusers_name in unet_conversion_map:
        mapping[diffusers_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, diffusers_part in unet_conversion_map_resnet:
                v = v.replace(diffusers_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, diffusers_part in unet_conversion_map_layer:
            v = v.replace(diffusers_part, sd_part)
        mapping[k] = v
    new_state_dict = {v: controlnet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict


def convert_controlnet_state_dict_to_diffusers(controlnet_state_dict):
    """
    Converts a Stable Diffusion LDM ControlNet state dictionary to a Diffusers format.

    Args:
        controlnet_state_dict (dict): The Stable Diffusion LDM ControlNet state dictionary.

    Returns:
        dict: The converted Diffusers state dictionary.
    """
    unet_conversion_map, unet_conversion_map_resnet, unet_conversion_map_layer = controlnet_conversion_map()

    mapping = {k: k for k in controlnet_state_dict}
    for sd_name, diffusers_name in unet_conversion_map:
        mapping[sd_name] = diffusers_name
    for k, v in mapping.items():
        for sd_part, diffusers_part in unet_conversion_map_layer:
            v = v.replace(sd_part, diffusers_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "resnets" in v:
            for sd_part, diffusers_part in unet_conversion_map_resnet:
                v = v.replace(sd_part, diffusers_part)
            mapping[k] = v
    new_state_dict = {v: controlnet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict


def load_checkpoint_with_text_encoder_conversion(ckpt_path, device="cpu"):
    """
    Loads a checkpoint and converts the text encoder state dict if necessary.
    Handles models where 'text_model' key is missing.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        device (str): Device to load the checkpoint onto.

    Returns:
        tuple: A tuple containing (checkpoint, state_dict).
    """
    # text encoderの格納形式が違うモデルに対応する ('text_model'がない)
    TEXT_ENCODER_KEY_REPLACEMENTS = [
        ("cond_stage_model.transformer.embeddings.", "cond_stage_model.transformer.text_model.embeddings."),
        ("cond_stage_model.transformer.encoder.", "cond_stage_model.transformer.text_model.encoder."),
        ("cond_stage_model.transformer.final_layer_norm.", "cond_stage_model.transformer.text_model.final_layer_norm."),
    ]

    if is_safetensors(ckpt_path):
        checkpoint = None
        state_dict = load_file(ckpt_path)  # , device) # may causes error
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            checkpoint = None

    key_reps = []
    for rep_from, rep_to in TEXT_ENCODER_KEY_REPLACEMENTS:
        for key in state_dict:
            if key.startswith(rep_from):
                new_key = rep_to + key[len(rep_from) :]
                key_reps.append((key, new_key))

    for key, new_key in key_reps:
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

    return checkpoint, state_dict


# TODO dtype指定の動作が怪しいので確認する text_encoderを指定形式で作れるか未確認
def load_models_from_stable_diffusion_checkpoint(v2, ckpt_path, device="cpu", dtype=None, unet_use_linear_projection_in_v2=True):
    """
    Loads text encoder, VAE, and U-Net from a Stable Diffusion checkpoint.

    Args:
        v2 (bool): Whether the model is SD v2.
        ckpt_path (str): Path to the checkpoint file.
        device (str): Device to load the models onto.
        dtype (torch.dtype, optional): Data type for loading the models.
        unet_use_linear_projection_in_v2 (bool): Whether to use linear projection in V2 U-Net.

    Returns:
        tuple: A tuple containing (text_model, vae, unet).
    """
    _, state_dict = load_checkpoint_with_text_encoder_conversion(ckpt_path, device)

    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(v2, unet_use_linear_projection_in_v2)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(v2, state_dict, unet_config)

    unet = UNet2DConditionModel(**unet_config).to(device)
    info = unet.load_state_dict(converted_unet_checkpoint)
    logger.info(f"loading u-net: {info}")

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config()
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)

    vae = AutoencoderKL(**vae_config).to(device)  # type: ignore[union-attr]
    info = vae.load_state_dict(converted_vae_checkpoint)
    logger.info(f"loading vae: {info}")

    # convert text_model
    if v2:
        converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v2(state_dict, 77)
        cfg = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=23,
            num_attention_heads=16,
            max_position_embeddings=77,
            hidden_act="gelu",
            layer_norm_eps=1e-05,
            dropout=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            model_type="clip_text_model",
            projection_dim=512,
            torch_dtype="float32",
            transformers_version="4.25.0.dev0",
        )
        text_model = CLIPTextModel._from_config(cfg)
        info = text_model.load_state_dict(converted_text_encoder_checkpoint)
    else:
        converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v1(state_dict)

        # logging.set_verbosity_error()  # don't show annoying warning
        # text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        # logging.set_verbosity_warning()
        # logger.info(f"config: {text_model.config}")
        cfg = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=77,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-05,
            dropout=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            model_type="clip_text_model",
            projection_dim=768,
            torch_dtype="float32",
        )
        text_model = CLIPTextModel._from_config(cfg)
        info = text_model.load_state_dict(converted_text_encoder_checkpoint)
    logger.info(f"loading text encoder: {info}")

    return text_model, vae, unet


def get_model_version_str_for_sd1_sd2(v2, v_parameterization):
    """
    Returns a string representing the model version (SD1, SD2, etc.).

    Args:
        v2 (bool): Whether the model is SD v2.
        v_parameterization (bool): Whether the model uses v-parameterization.

    Returns:
        str: The model version string.
    """
    # only for reference
    version_str = "sd"
    if v2:
        version_str += "_v2"
    else:
        version_str += "_v1"
    if v_parameterization:
        version_str += "_v"
    return version_str


def convert_text_encoder_state_dict_to_sd_v2(checkpoint, make_dummy_weights=False):
    """
    Converts a SD v2 Text Encoder state dictionary to the Stable Diffusion format.

    Args:
        checkpoint (dict): The Diffusers text encoder state dictionary.
        make_dummy_weights (bool): Whether to create dummy weights for missing keys.

    Returns:
        dict: The converted Stable Diffusion state dictionary.
    """

    def convert_key(key):
        # position_idsの除去
        if ".position_ids" in key:
            return None

        # common
        key = key.replace("text_model.encoder.", "transformer.")
        key = key.replace("text_model.", "")
        if "layers" in key:
            # resblocks conversion
            key = key.replace(".layers.", ".resblocks.")
            if ".layer_norm" in key:
                key = key.replace(".layer_norm", ".ln_")
            elif ".mlp." in key:
                key = key.replace(".fc1.", ".c_fc.")
                key = key.replace(".fc2.", ".c_proj.")
            elif ".self_attn.out_proj" in key:
                key = key.replace(".self_attn.out_proj.", ".attn.out_proj.")
            elif ".self_attn." in key:
                key = None  # 特殊なので後で処理する
            else:
                raise ValueError(f"unexpected key in DiffUsers model: {key}")
        elif ".position_embedding" in key:
            key = key.replace("embeddings.position_embedding.weight", "positional_embedding")
        elif ".token_embedding" in key:
            key = key.replace("embeddings.token_embedding.weight", "token_embedding.weight")
        elif "final_layer_norm" in key:
            key = key.replace("final_layer_norm", "ln_final")
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if "layers" in key and "q_proj" in key:
            # 三つを結合
            key_q = key
            key_k = key.replace("q_proj", "k_proj")
            key_v = key.replace("q_proj", "v_proj")

            value_q = checkpoint[key_q]
            value_k = checkpoint[key_k]
            value_v = checkpoint[key_v]
            value = torch.cat([value_q, value_k, value_v])

            new_key = key.replace("text_model.encoder.layers.", "transformer.resblocks.")
            new_key = new_key.replace(".self_attn.q_proj.", ".attn.in_proj_")
            new_sd[new_key] = value

    # 最後の層などを捏造するか
    if make_dummy_weights:
        logger.info("make dummy weights for resblock.23, text_projection and logit scale.")
        keys = list(new_sd.keys())
        for key in keys:
            if key.startswith("transformer.resblocks.22."):
                new_sd[key.replace(".22.", ".23.")] = new_sd[key].clone()  # copyしないとsafetensorsの保存で落ちる

        # Diffusersに含まれない重みを作っておく
        new_sd["text_projection"] = torch.ones((1024, 1024), dtype=new_sd[keys[0]].dtype, device=new_sd[keys[0]].device)
        new_sd["logit_scale"] = torch.tensor(1)

    return new_sd


def save_stable_diffusion_checkpoint(v2, output_file, text_encoder, unet, ckpt_path, epochs, steps, metadata, save_dtype=None, vae=None):
    """
    Saves a Stable Diffusion checkpoint.

    Args:
        v2 (bool): Whether the model is SD v2.
        output_file (str): Path to the output file.
        text_encoder (CLIPTextModel): The text encoder model.
        unet (UNet2DConditionModel): The U-Net model.
        ckpt_path (str): Path to the original checkpoint (used for reference).
        epochs (int): Number of epochs trained.
        steps (int): Number of global steps trained.
        metadata (dict): Metadata to save with the checkpoint.
        save_dtype (torch.dtype, optional): Data type to save the checkpoint in.
        vae (AutoencoderKL, optional): The VAE model.

    Returns:
        int: The number of keys in the saved state dictionary.
    """
    if ckpt_path is not None:
        # epoch/stepを参照する。またVAEがメモリ上にないときなど、もう一度VAEを含めて読み込む
        checkpoint, state_dict = load_checkpoint_with_text_encoder_conversion(ckpt_path)
        if checkpoint is None:  # safetensors または state_dictのckpt
            checkpoint = {}
            strict = False
        else:
            strict = True
        if "state_dict" in state_dict:
            del state_dict["state_dict"]
    else:
        # 新しく作る
        assert vae is not None, "VAE is required to save a checkpoint without a given checkpoint"
        checkpoint = {}
        state_dict = {}
        strict = False

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            assert not strict or key in state_dict, f"Illegal key in save SD: {key}"
            if save_dtype is not None:
                v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    # Convert the UNet model
    unet_state_dict = convert_unet_state_dict_to_sd(v2, unet.state_dict())
    update_sd("model.diffusion_model.", unet_state_dict)

    # Convert the text encoder model
    if v2:
        make_dummy = ckpt_path is None  # 参照元のcheckpointがない場合は最後の層を前の層から複製して作るなどダミーの重みを入れる
        text_enc_dict = convert_text_encoder_state_dict_to_sd_v2(text_encoder.state_dict(), make_dummy)
        update_sd("cond_stage_model.model.", text_enc_dict)
    else:
        text_enc_dict = text_encoder.state_dict()
        update_sd("cond_stage_model.transformer.", text_enc_dict)

    # Convert the VAE
    if vae is not None:
        vae_dict = convert_vae_state_dict(vae.state_dict())
        update_sd("first_stage_model.", vae_dict)

    # Put together new checkpoint
    key_count = len(state_dict.keys())
    new_ckpt = {"state_dict": state_dict}

    # epoch and global_step are sometimes not int
    try:
        if "epoch" in checkpoint:
            epochs += checkpoint["epoch"]
        if "global_step" in checkpoint:
            steps += checkpoint["global_step"]
    except Exception:
        pass

    new_ckpt["epoch"] = epochs
    new_ckpt["global_step"] = steps

    if is_safetensors(output_file):
        # TODO Tensor以外のdictの値を削除したほうがいいか
        save_file(state_dict, output_file, metadata)
    else:
        torch.save(new_ckpt, output_file)

    return key_count


def save_diffusers_checkpoint(v2, output_dir, text_encoder, unet, pretrained_model_name_or_path, vae=None, use_safetensors=False):
    """
    Saves a Diffusers checkpoint.

    Args:
        v2 (bool): Whether the model is SD v2.
        output_dir (str): Directory to save the checkpoint.
        text_encoder (CLIPTextModel): The text encoder model.
        unet (UNet2DConditionModel): The U-Net model.
        pretrained_model_name_or_path (str): Path to the pretrained model or model ID.
        vae (AutoencoderKL, optional): The VAE model.
        use_safetensors (bool): Whether to use safetensors format.
    """
    if pretrained_model_name_or_path is None:
        # load default settings for v1/v2
        if v2:
            pretrained_model_name_or_path = DIFFUSERS_REF_MODEL_ID_V2
        else:
            pretrained_model_name_or_path = DIFFUSERS_REF_MODEL_ID_V1

    scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    if vae is None:
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

    # original U-Net cannot be saved, so we need to convert it to the Diffusers version
    # TODO this consumes a lot of memory
    diffusers_unet = diffusers.UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    diffusers_unet.load_state_dict(unet.state_dict())

    pipeline = StableDiffusionPipeline(
        unet=diffusers_unet,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        tokenizer=tokenizer,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=None,
    )
    pipeline.save_pretrained(output_dir, safe_serialization=use_safetensors)  # type: ignore[union-attr]


def create_unet_diffusers_config(v2, use_linear_projection_in_v2=False):
    """
    Creates a config for the diffusers UNet based on the LDM model.

    Args:
        v2 (bool): Whether the model is SD v2.
        use_linear_projection_in_v2 (bool): Whether to use linear projection in V2.

    Returns:
        dict: The UNet configuration dictionary.
    """
    # unet_params = original_config.model.params.unet_config.params

    block_out_channels = [UNET_PARAMS_MODEL_CHANNELS * mult for mult in UNET_PARAMS_CHANNEL_MULT]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for _i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    config = {
        "sample_size": UNET_PARAMS_IMAGE_SIZE,
        "in_channels": UNET_PARAMS_IN_CHANNELS,
        "out_channels": UNET_PARAMS_OUT_CHANNELS,
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": UNET_PARAMS_NUM_RES_BLOCKS,
        "cross_attention_dim": UNET_PARAMS_CONTEXT_DIM if not v2 else V2_UNET_PARAMS_CONTEXT_DIM,
        "attention_head_dim": UNET_PARAMS_NUM_HEADS if not v2 else V2_UNET_PARAMS_ATTENTION_HEAD_DIM,
        # use_linear_projection=UNET_PARAMS_USE_LINEAR_PROJECTION if not v2 else V2_UNET_PARAMS_USE_LINEAR_PROJECTION,
    }
    if v2 and use_linear_projection_in_v2:
        config["use_linear_projection"] = True

    return config


def convert_ldm_unet_checkpoint(v2, checkpoint, config):
    """
    Converts an LDM UNet checkpoint to the Diffusers format.

    Args:
        v2 (bool): Whether the model is SD v2.
        checkpoint (dict): The LDM checkpoint state dictionary.
        config (dict): The Diffusers UNet configuration.

    Returns:
        dict: The converted Diffusers UNet state dictionary.
    """

    # extract state_dict for UNet
    unet_state_dict = {}
    unet_key = "model.diffusion_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(unet_key):
            unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}." in key] for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}." in key] for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}." in key] for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(f"input_blocks.{i}.0.op.weight")
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(f"input_blocks.{i}.0.op.bias")

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)

            # オリジナル：
            # if ["conv.weight", "conv.bias"] in output_block_list.values():
            #   index = list(output_block_list.values()).index(["conv.weight", "conv.bias"])

            # biasとweightの順番に依存しないようにする：もっといいやり方がありそうだが
            for l in output_block_list.values():
                l.sort()

            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[f"output_blocks.{i}.{index}.conv.bias"]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[f"output_blocks.{i}.{index}.conv.weight"]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config)
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    # SDのv2では1*1のconv2dがlinearに変わっている
    # 誤って Diffusers 側を conv2d のままにしてしまったので、変換必要
    if v2 and not config.get("use_linear_projection", False):
        linear_transformer_to_conv(new_checkpoint)

    return new_checkpoint


def linear_transformer_to_conv(checkpoint):
    """
    Converts linear weights to 1x1 convolutional weights in transformers.

    Args:
        checkpoint (dict): The state dictionary to modify in-place.
    """
    keys = list(checkpoint.keys())
    tf_keys = ["proj_in.weight", "proj_out.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in tf_keys and checkpoint[key].ndim == 2:
            checkpoint[key] = checkpoint[key].unsqueeze(2).unsqueeze(2)
