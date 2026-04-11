import math
from typing import Any

import torch
from torch import distributed as dist
from torch import nn

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss


class AdamMini(BaseOptimizer):  # pragma: no cover
    r"""Use fewer learning rates to gain more.

    This repo-owned adaptation keeps the donor update logic, but accepts the
    shared optimizer-factory parameter-group payload instead of requiring a
    raw `nn.Module` instance. When named parameter groups are provided, the
    optimizer retains the donor name-based grouping behavior.

    :param model_or_params: nn.Module or optimizer parameter groups.
    :param model_sharding: bool. set to True if you are using model
        parallelism with more than 1 GPU, including FSDP and zero_1, 2, 3 in
        Deepspeed. Set to False otherwise.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of
        gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param num_embeds: int. number of embedding dimensions.
    :param num_heads: int. number of attention heads.
    :param num_query_groups: int. number of query groups in GQA.
    :param eps: float. term added to the denominator to improve numerical
        stability.
    """

    def __init__(
        self,
        model_or_params,
        lr: float = 1.0,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.1,
        model_sharding: bool = False,
        num_embeds: int = 2048,
        num_heads: int = 32,
        num_query_groups: int | None = None,
        eps: float = 1e-8,
        **kwargs,
    ):
        del kwargs

        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, "weight_decay")
        self.validate_non_negative(num_embeds, "num_embeds")
        self.validate_non_negative(num_heads, "num_heads")
        self.validate_non_negative(eps, "eps")

        self.num_query_groups = num_query_groups if num_query_groups is not None else num_embeds
        self.validate_mod(num_embeds, self.num_query_groups)

        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else max(torch.cuda.device_count(), 1)

        self.model = model_or_params if isinstance(model_or_params, nn.Module) else None
        self.model_sharding = model_sharding
        self.num_embeds = num_embeds
        self.num_heads = num_heads

        self.embed_blocks = {"embed", "embd", "wte", "lm_head.weight", "output.weight"}
        self.qk_blocks = {"k_proj.weight", "q_proj.weight", "wq.weight", "wk.weight"}

        groups = self.get_optimizer_groups(model_or_params, weight_decay)

        defaults: Defaults = {"lr": lr, "betas": betas, "eps": eps}
        super().__init__(groups, defaults)

    def __str__(self) -> str:
        return "AdamMini"

    def init_group(self, group, **kwargs) -> None:
        del group, kwargs

    def _iter_named_parameters(self, model_or_params) -> list[tuple[str, torch.nn.Parameter, dict[str, Any]]]:
        if isinstance(model_or_params, nn.Module):
            return [(name, param, {}) for name, param in model_or_params.named_parameters() if param.requires_grad]

        named_entries: list[tuple[str, torch.nn.Parameter, dict[str, Any]]] = []
        if isinstance(model_or_params, list) and model_or_params and isinstance(model_or_params[0], dict):
            for group_index, group in enumerate(model_or_params):
                group_options = {key: value for key, value in group.items() if key not in {"params", "named_params"}}
                params = group["params"]
                if isinstance(params, torch.Tensor):
                    params = [params]
                else:
                    params = list(params)

                named_params = group.get("named_params")
                if named_params is not None:
                    name_by_id = {id(param): name for name, param in named_params}
                    for param_index, param in enumerate(params):
                        if not param.requires_grad:
                            continue
                        name = name_by_id.get(id(param), f"group{group_index}.param{param_index}")
                        named_entries.append((name, param, group_options))
                    continue

                label = group.get("label", f"group{group_index}")
                for param_index, param in enumerate(params):
                    if not param.requires_grad:
                        continue
                    named_entries.append((f"{label}.param{param_index}", param, group_options))
            return named_entries

        params = model_or_params if isinstance(model_or_params, list) else list(model_or_params)
        for index, param in enumerate(params):
            if not param.requires_grad:
                continue
            named_entries.append((f"param_{index}", param, {}))
        return named_entries

    def get_optimizer_groups(self, model_or_params, weight_decay: float):
        groups = []
        for name, param, group_options in self._iter_named_parameters(model_or_params):
            group_lr = group_options.get("lr")
            group_weight_decay = group_options.get("weight_decay", weight_decay)

            group = {
                "name": name,
                "params": [param],
                "weight_decay": 0.0 if ("norm" in name or "ln_f" in name) else group_weight_decay,
            }
            if group_lr is not None:
                group["lr"] = group_lr
            if "eps" in group_options:
                group["eps"] = group_options["eps"]
            if "betas" in group_options:
                group["betas"] = group_options["betas"]

            if any(block in name for block in self.qk_blocks):
                group["parameter_per_head"] = self.num_embeds * self.num_embeds // self.num_heads

            if "attn.attn.weight" in name or "attn.qkv.weight" in name:
                group["n_head"] = self.num_heads
                group["q_per_kv"] = self.num_embeds // self.num_query_groups

            groups.append(group)

        return groups

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group["step"] = 0
            for parameter in group["params"]:
                state = self.state[parameter]
                state["m"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["v"] = torch.zeros_like(parameter, dtype=torch.float32)

    @staticmethod
    def step_embed(
        parameter,
        grad,
        state,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            state["m"] = torch.zeros_like(parameter, dtype=torch.float32)
            state["v"] = torch.zeros_like(parameter, dtype=torch.float32)

        m, v = state["m"], state["v"]
        m.lerp_(grad, weight=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad.conj(), value=1.0 - beta2)
        denom = (v.sqrt() / bias_correction2_sq).add_(eps)
        parameter.addcdiv_(m, denom, value=-lr / bias_correction1)

    @staticmethod
    def step_attn_proj(
        parameter,
        grad,
        state,
        parameter_per_head: int,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            state["m"] = torch.zeros_like(parameter, dtype=torch.float32).view(-1, parameter_per_head)
            state["head"] = state["m"].shape[0]
            state["v_mean"] = torch.zeros(state["head"], device=state["m"].device)

        m, v = state["m"], state["v_mean"]
        head = state["head"]
        grad = grad.view(head, parameter_per_head)
        m.lerp_(grad, weight=1.0 - beta1)
        tmp_lr = torch.mean(grad * grad, dim=1).to(m.device)
        v.mul_(beta2).add_(tmp_lr, alpha=1.0 - beta2)
        denom = (v.sqrt() / bias_correction2_sq).add_(eps)
        update = (1 / (denom * bias_correction1)).view(head, 1) * m
        update = update.view_as(parameter)
        parameter.add_(update, alpha=-lr)

    @staticmethod
    def step_attn(
        parameter,
        grad,
        state,
        num_heads: int,
        q_per_kv: int,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            state["m"] = torch.zeros_like(parameter, dtype=torch.float32).view(num_heads, q_per_kv + 2, -1)
            state["v_mean"] = torch.zeros(num_heads, q_per_kv + 2, device=state["m"].device)

        m, v = state["m"], state["v_mean"]
        grad = grad.view(num_heads, q_per_kv + 2, -1)
        m.lerp_(grad, weight=1.0 - beta1)
        tmp_lr = torch.mean(grad * grad, dim=2).to(m.device)
        v.mul_(beta2).add_(tmp_lr, alpha=1.0 - beta2)
        denom = (v.sqrt() / bias_correction2_sq).add_(eps)
        update = (1 / (denom * bias_correction1)).view(num_heads, q_per_kv + 2, -1) * m
        update = update.view_as(parameter)
        parameter.add_(update, alpha=-lr)

    def step_lefts(
        self,
        parameter,
        grad,
        state,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            dim = torch.tensor(parameter.numel(), device=parameter.device, dtype=torch.float32)
            reduced = False
            if self.model_sharding and self.world_size > 1 and dist.is_available() and dist.is_initialized():
                tensor_list = [torch.zeros_like(dim) for _ in range(self.world_size)]
                dist.all_gather(tensor_list, dim)

                shard_count = 0
                dim = torch.tensor(0.0, device=dim.device)
                for gathered_dim in tensor_list:
                    if gathered_dim > 0:
                        shard_count += 1
                    dim += gathered_dim
                if shard_count >= 2:
                    reduced = True

            state["m"] = torch.zeros_like(parameter, dtype=torch.float32)
            state["v_mean"] = torch.tensor(0.0, device=parameter.device, dtype=torch.float32)
            state["dimension"] = dim
            state["reduced"] = reduced

        tmp_lr = torch.sum(grad * grad)
        if state["reduced"] and dist.is_available() and dist.is_initialized():
            dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)
        tmp_lr.div_(state["dimension"])

        m, v = state["m"], state["v_mean"]
        m.lerp_(grad, weight=1.0 - beta1)
        v.mul_(beta2).add_(tmp_lr, alpha=1.0 - beta2)
        denom = (v.sqrt() / bias_correction2_sq).add_(eps)
        step_size = (1 / bias_correction1) / denom
        update = m * step_size
        parameter.add_(update, alpha=-lr)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            name = group["name"]
            beta1, beta2 = group["betas"]
            bias_correction1 = self.debias(beta1, group["step"])
            bias_correction2 = self.debias(beta2, group["step"])
            bias_correction2_sq = math.sqrt(bias_correction2)

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                grad = grad.to(torch.float32)
                state = self.state[parameter]

                self.apply_weight_decay(
                    p=parameter,
                    grad=grad,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    weight_decouple=True,
                    fixed_decay=False,
                )

                if any(block in name for block in self.embed_blocks):
                    self.step_embed(parameter, grad, state, group["lr"], beta1, beta2, bias_correction1, bias_correction2_sq, group["eps"])
                elif any(block in name for block in self.qk_blocks):
                    self.step_attn_proj(
                        parameter,
                        grad,
                        state,
                        group["parameter_per_head"],
                        group["lr"],
                        beta1,
                        beta2,
                        bias_correction1,
                        bias_correction2_sq,
                        group["eps"],
                    )
                elif "attn.attn.weight" in name or "attn.qkv.weight" in name:
                    self.step_attn(
                        parameter,
                        grad,
                        state,
                        group["n_head"],
                        group["q_per_kv"],
                        group["lr"],
                        beta1,
                        beta2,
                        bias_correction1,
                        bias_correction2_sq,
                        group["eps"],
                    )
                else:
                    self.step_lefts(parameter, grad, state, group["lr"], beta1, beta2, bias_correction1, bias_correction2_sq, group["eps"])

        return loss
