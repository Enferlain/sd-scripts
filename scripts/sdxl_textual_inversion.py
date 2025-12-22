import hydra
import os
import torch

from typing import Optional, Union

import sd_textual_inversion

from library.constants import VAE_SCALE_FACTOR, MODEL_VERSION_SDXL_BASE_V1_0
from library.models.sdxl_model_util import get_size_embeddings
from library.utils.device_utils import init_ipex
from library.strategies import strategy_sdxl, strategy_sd
from library.data.dataset import DatasetGroup, MinimalDataset
from library.training.sdxl_sample_generation import sample_images
from library.training.sdxl_model_prep import load_target_model as load_target_model_sdxl
from library.config.dataclasses.sdxl_textual_inversion import SDXLTextualInversionConfig

init_ipex()


class SdxlTextualInversionTrainer(sd_textual_inversion.TextualInversionTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = VAE_SCALE_FACTOR
        self.is_sdxl = True

    def assert_extra_args(self, args, train_dataset_group: Union[DatasetGroup, MinimalDataset], val_dataset_group: Optional[
        DatasetGroup]):
        # For SDXL, we just verify bucket resolution
        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, model_config, performance_config, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = load_target_model_sdxl(model_config, accelerator, MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        return MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def get_tokenize_strategy(self, model_config, training_config):
        return strategy_sdxl.SdxlTokenizeStrategy(training_config.max_token_length, model_config.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sdxl.SdxlTokenizeStrategy):
        return [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

    def get_latents_caching_strategy(self, dataset_config):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            False, dataset_config.cache_latents_to_disk, dataset_config.vae_batch_size, dataset_config.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, training_config):
        return strategy_sdxl.SdxlTextEncodingStrategy()

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def sample_images(
        self, accelerator, sampling_config, training_config, saving_config, epoch, global_step, device, vae, tokenizers, text_encoders, unet, prompt_replacement
    ):
        sample_images(
            accelerator, sampling_config, training_config, saving_config, epoch, global_step, device, vae, tokenizers, text_encoders, unet, prompt_replacement
        )

    def save_weights(self, file, updated_embs, save_dtype, metadata):
        state_dict = {"clip_l": updated_embs[0], "clip_g": updated_embs[1]}

        if save_dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(save_dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            data = load_file(file)
        else:
            data = torch.load(file, map_location="cpu")

        emb_l = data.get("clip_l", None)  # ViT-L text encoder 1
        emb_g = data.get("clip_g", None)  # BiG-G text encoder 2

        assert (
            emb_l is not None or emb_g is not None
        ), f"weight file does not contain weights for text encoder 1 or 2: {file}"

        return [emb_l, emb_g]


@hydra.main(config_path="../configs", config_name="sdxl_textual_inversion", version_base=None)
def main(config: SDXLTextualInversionConfig):
    trainer = SdxlTextualInversionTrainer()
    trainer.train(config)


if __name__ == "__main__":
    main()
