import hydra
import os
import torch

import library.strategies.sd.caching
import library.strategies.sdxl.encoding
import library.strategies.sdxl.tokenization
from scripts._deprecated import sd_textual_inversion

from library.constants import SDXL_VAE_LATENT_SCALE, MODEL_VERSION_SDXL_BASE_V1_0
from library.models.sdxl.conversion import get_size_embeddings
from library.utils.device_utils import init_ipex
from library.data._deprecated.dataset import DatasetGroup, MinimalDataset
from library.training._deprecated.sdxl_sample_generation import sample_images
from library.models.sdxl.loader import load_target_model as load_target_model_sdxl
from library.config.dataclasses.run import RunConfig
from library.config.config_validation import prepare_config, validate_config, validate_dataset_groups
from library.config.schemas import register_run

init_ipex()


# TODO: Proper training loop needs to be implemented (not parented to sd_textual_inversion)
class SdxlTextualInversionTrainer(sd_textual_inversion.TextualInversionTrainer):
    def __init__(self):
        super().__init__()
        self.vae_latent_scale = SDXL_VAE_LATENT_SCALE
        self.is_sdxl = True

    def validate_extra_config(self, config, train_dataset_group: DatasetGroup | MinimalDataset, val_dataset_group: DatasetGroup | None):
        validate_dataset_groups(config, train_dataset_group, val_dataset_group)

    def load_target_model(self, cfg, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = load_target_model_sdxl(
            cfg.model,
            cfg.performance.memory,
            cfg.data.caching,
            cfg.performance.precision,
            accelerator,
            MODEL_VERSION_SDXL_BASE_V1_0,
            weight_dtype,
        )

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        return MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def get_tokenize_strategy(self, cfg):
        return library.strategies.sdxl.tokenization.SdxlTokenizeStrategy(cfg.training.max_token_length, cfg.model.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: library.strategies.sdxl.tokenization.SdxlTokenizeStrategy):
        return [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

    def get_text_encoding_strategy(self, cfg):
        return library.strategies.sdxl.encoding.SdxlTextEncodingStrategy()

    def call_unet(self, config, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
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
        self,
        accelerator,
        sampling_config,
        training_config,
        saving_config,
        loss_config,
        epoch,
        global_step,
        device,
        vae,
        tokenizers,
        text_encoders,
        unet,
        prompt_replacement,
    ):
        sample_images(
            accelerator,
            sampling_config,
            training_config,
            saving_config,
            loss_config,
            epoch,
            global_step,
            device,
            vae,
            tokenizers,
            text_encoders,
            unet,
            prompt_replacement,
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

        assert emb_l is not None or emb_g is not None, f"weight file does not contain weights for text encoder 1 or 2: {file}"

        return [emb_l, emb_g]


register_run()


@hydra.main(config_path="../configs", config_name="presets/sdxl_textual_inversion", version_base=None)
def main(config: RunConfig):
    prepare_config(config)
    validate_config(config)
    trainer = SdxlTextualInversionTrainer()
    trainer.train(config)


if __name__ == "__main__":
    main()
