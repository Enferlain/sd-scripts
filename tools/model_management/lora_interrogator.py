from tqdm import tqdm
import torch
import logging
import argparse

from transformers import CLIPTokenizer

from library.constants import V2_STABLE_DIFFUSION_ID, TOKENIZER_ID
from library.networks import lora
from library.utils.common_utils import setup_logging
from library.utils.device_utils import init_ipex, get_preferred_device
from library.training.model_prep import _load_target_model

init_ipex()

setup_logging()
logger = logging.getLogger(__name__)

DEVICE = get_preferred_device()


def interrogate(args):
  weights_dtype = torch.float16

  # Prepare various things
  logger.info(f"loading SD model: {args.sd_model}")
  args.pretrained_model_name_or_path = args.sd_model
  args.vae = None
  text_encoder, vae, unet, _ = _load_target_model(args, weights_dtype, DEVICE)

  logger.info(f"loading LoRA: {args.model}")
  network, weights_sd = lora.create_network_from_weights(1.0, args.model, vae, text_encoder, unet)

  # Check if there are weights for text encoder: Actually it's better to do this on the lora side
  has_te_weight = False
  for key in weights_sd.keys():
    if 'lora_te' in key:
      has_te_weight = True
      break
  if not has_te_weight:
    logger.error("This LoRA does not have modules for Text Encoder, cannot interrogate")
    return
  del vae

  logger.info("loading tokenizer")
  if args.v2:
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(V2_STABLE_DIFFUSION_ID, subfolder="tokenizer")
  else:
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_ID)  # , model_max_length=max_token_length + 2)

  text_encoder.to(DEVICE, dtype=weights_dtype)
  text_encoder.eval()
  unet.to(DEVICE, dtype=weights_dtype)
  unet.eval()               # U-Net is not called so not needed

  # Hit tokens one by one
  token_id_start = 0
  token_id_end = max(tokenizer.all_special_ids)
  logger.info(f"interrogate tokens are: {token_id_start} to {token_id_end}")

  def get_all_embeddings(text_encoder):
    embs = []
    with torch.no_grad():
      for token_id in tqdm(range(token_id_start, token_id_end + 1, args.batch_size)):
        batch = []
        for tid in range(token_id, min(token_id_end + 1, token_id + args.batch_size)):
          tokens = [tokenizer.bos_token_id, tid, tokenizer.eos_token_id]
          # tokens = [tid]                                                    # This result is not so good
          batch.append(tokens)

        # batch_embs = text_encoder(torch.tensor(batch).to(DEVICE))[0].to("cpu")  # Including bos/eos seems to make a difference [:, 1]
        # Support clip skip
        batch = torch.tensor(batch).to(DEVICE)
        if args.clip_skip is None:
          encoder_hidden_states = text_encoder(batch)[0]
        else:
          enc_out = text_encoder(batch, output_hidden_states=True, return_dict=True)
          encoder_hidden_states = enc_out['hidden_states'][-args.clip_skip]
          encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.to("cpu")

        embs.extend(encoder_hidden_states)
    return torch.stack(embs)

  logger.info("get original text encoder embeddings.")
  orig_embs = get_all_embeddings(text_encoder)

  network.apply_to(text_encoder, unet, True, len(network.unet_loras) > 0)
  info = network.load_state_dict(weights_sd, strict=False)
  logger.info(f"Loading LoRA weights: {info}")

  network.to(DEVICE, dtype=weights_dtype)
  network.eval()

  del unet

  logger.info("You can ignore warning messages start with '_IncompatibleKeys' (LoRA model does not have alpha because trained by older script)")
  logger.info("get text encoder embeddings with lora.")
  lora_embs = get_all_embeddings(text_encoder)

  # Compare: simply by absolute difference for now
  logger.info("comparing...")
  diffs = {}
  for i, (orig_emb, lora_emb) in enumerate(zip(orig_embs, tqdm(lora_embs))):
    diff = torch.mean(torch.abs(orig_emb - lora_emb))
    # diff = torch.mean(torch.cosine_similarity(orig_emb, lora_emb, dim=1))       # Cannot detect well
    diff = float(diff.detach().to('cpu').numpy())
    diffs[token_id_start + i] = diff

  diffs_sorted = sorted(diffs.items(), key=lambda x: -x[1])

  # Display results
  print("top 100:")
  for i, (token, diff) in enumerate(diffs_sorted[:100]):
    # if diff < 1e-6:
    #   break
    string = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([token]))
    print(f"[{i:3d}]: {token:5d} {string:<20s}: {diff:.5f}")


def setup_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()

  parser.add_argument("--v2", action='store_true',
                      help='load Stable Diffusion v2.x model')
  parser.add_argument("--sd_model", type=str, default=None,
                      help="Stable Diffusion model to load: ckpt or safetensors file")
  parser.add_argument("--model", type=str, default=None,
                      help="LoRA model to interrogate: ckpt or safetensors file")
  parser.add_argument("--batch_size", type=int, default=16,
                      help="batch size for processing with Text Encoder")
  parser.add_argument("--clip_skip", type=int, default=None,
                      help="use output of nth layer from back of text encoder (n>=1)")

  return parser


if __name__ == '__main__':
  parser = setup_parser()

  args = parser.parse_args()
  interrogate(args)
