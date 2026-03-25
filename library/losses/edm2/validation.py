from __future__ import annotations

import logging

from library.config.dataclasses.loss import EDM2Config

logger = logging.getLogger(__name__)


def handle_conflicting_configuration(edm2_config: EDM2Config, snr_config=None):
    """
    Resolve configuration conflicts between EDM2 importance weighting and SNR-based loss options.

    When ``snr_config`` is omitted, this helper falls back to the older flat-config shape
    where the SNR settings live directly on ``edm2_config``.
    """
    snr_config = edm2_config if snr_config is None else snr_config

    if edm2_config.enabled and edm2_config.importance.enabled and not edm2_config.importance.safety_override:
        if getattr(snr_config, "debiased_estimation_loss", False):
            snr_config.debiased_estimation_loss = False
            logger.warning(
                "Debiased estimation loss AND EDM2 loss weighting with importance weighting are enabled. "
                "It is not advised to use both, as there is a possibility of loss curving to 0 as SNR approaches 0, "
                "as such, **Debiased estimation loss has been DISABLED**. "
                "You may override this behavior by setting loss.edm2.importance.safety_override=true."
            )

        if getattr(snr_config, "min_snr_gamma", None):
            logger.warning(
                "Min snr gamma AND EDM2 loss weighting with importance weighting are enabled. "
                "It is not advised to use both, as there is a possibility of loss curving to 0 as SNR approaches 0, "
                "as such, **min snr gamma has been DISABLED**. "
                "You may override this behavior by setting loss.edm2.importance.safety_override=true."
            )
            snr_config.min_snr_gamma = None
