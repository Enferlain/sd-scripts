"""
SD3 upstream donor status for ``sd3_models.py``.

Fully cut into the repo:
- ``library/models/sd3/mmdit.py``
- ``library/models/sd3/vae.py``

Ported symbols included:
- ``SD3Params``
- positional embedding helpers
- ``PatchEmbed`` / ``UnPatch`` / ``MLP`` / ``TimestepEmbedding`` / ``Embedder``
- ``rmsnorm`` / ``RMSNorm`` / ``SwiGLUFeedForward`` / ``AttentionLinears``
- ``vanilla_attention`` / ``attention``
- ``SingleDiTBlock`` / ``MMDiTBlock`` / ``MMDiT`` / ``create_sd3_mmdit``
- ``Normalize`` / ``ResnetBlock`` / ``AttnBlock`` / ``Downsample`` / ``Upsample``
- ``VAEEncoder`` / ``VAEDecoder`` / ``SDVAE``

No remaining unported code is tracked from this donor file.
"""
