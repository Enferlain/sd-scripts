import bitsandbytes
import bitsandbytes.functional as F
import torch

from library.optimization.optimizers.stochastic import copy_stochastic_


class AdamW8bitKahan(bitsandbytes.optim.AdamW8bit):
    """AdamW8bit with a Kahan-style compensation buffer and optional stabilization."""

    def __init__(self, *args, stabilize: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.stabilize = stabilize

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        super().init_state(group, p, gindex, pindex)
        self.state[p]["shift"] = self.get_state_buffer(p, dtype=p.dtype)

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        p.data = p.data.contiguous()
        p.grad = p.grad.contiguous()

        state = self.state[p]
        grad = p.grad
        config = self.get_config(gindex, pindex, group)

        state["step"] += 1
        step = state["step"]

        if config["percentile_clipping"] < 100:
            _current_gnorm, _clip_value, gnorm_scale = F.percentile_clipping(
                grad,
                state["gnorm_vec"],
                step,
                config["percentile_clipping"],
            )
        else:
            gnorm_scale = 1.0

        shift = state["shift"]

        if self.stabilize:
            exp_avg_sq = state["state2"]
            eps_sq = torch.tensor(config["eps"] ** 2, dtype=exp_avg_sq.dtype, device=exp_avg_sq.device)
            rms = grad.pow(2).div_(exp_avg_sq.maximum(eps_sq)).mean().sqrt()
            lr = config["lr"] / max(1, rms.item())
        else:
            lr = config["lr"]

        if state["state1"].dtype == torch.float:
            F.optimizer_update_32bit(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                config["betas"][0],
                config["eps"],
                step,
                lr,
                state["state2"],
                config["betas"][1],
                config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                config["alpha"],
                0.0,
                gnorm_scale,
                state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )
        elif state["state1"].dtype == torch.uint8 and not config["block_wise"]:
            F.optimizer_update_8bit(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["eps"],
                step,
                lr,
                state["qmap1"],
                state["qmap2"],
                state["max1"],
                state["max2"],
                state["new_max1"],
                state["new_max2"],
                0.0,
                gnorm_scale=gnorm_scale,
                unorm_vec=state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
            )
            state["max1"], state["new_max1"] = state["new_max1"], state["max1"]
            state["max2"], state["new_max2"] = state["new_max2"], state["max2"]
        elif state["state1"].dtype == torch.uint8 and config["block_wise"]:
            F.optimizer_update_8bit_blockwise(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                config["alpha"],
                config["eps"],
                step,
                lr,
                state["qmap1"],
                state["qmap2"],
                state["absmax1"],
                state["absmax2"],
                0.0,
                gnorm_scale=gnorm_scale,
                skip_zeros=config["skip_zeros"],
            )

        weight_decay = config["weight_decay"]
        if weight_decay > 0.0:
            wd_update = p.data.float().mul_(lr * weight_decay)
            shift_fp32 = shift.float().sub_(wd_update)
            if shift.dtype == torch.bfloat16:
                copy_stochastic_(shift, shift_fp32)
            else:
                shift.copy_(shift_fp32)

        buffer = p.clone()
        p.add_(shift)
        shift.add_(buffer.sub_(p))
