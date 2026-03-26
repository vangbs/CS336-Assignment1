import torch
import math
from collections.abc import Iterable

class MyAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        **kwargs
    ):
        if kwargs.get("lr", -1) < 0:
            raise ValueError("Invalid learning rate")
        super().__init__(params, defaults=kwargs)
    
    @torch.no_grad()
    def step(
        self,
        _closure = None
    ):
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.to(torch.float32)
                m = state.get("m", torch.zeros_like(p, dtype=torch.float32))
                v = state.get("v", torch.zeros_like(p, dtype=torch.float32))
                t = state.get("t", 0)
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * (grad ** 2)
                t += 1
                alpha_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                
                p.sub_(m / (torch.sqrt(v) + eps), alpha=alpha_t)
                p.sub_(p, alpha=lr*weight_decay)
                
                state["m"] = m
                state["v"] = v
                state["t"] = t

        return None

class My_Cosine_Schedule:
    def __init__(
        self,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def get_learning_rate(
        self,
        it: int,
    ) -> float:
        if it < self.warmup_iters:
            return it / self.warmup_iters * self.max_learning_rate
        elif it > self.cosine_cycle_iters:
            return self.min_learning_rate
        else:
            return self.min_learning_rate + 0.5 * (
                self.max_learning_rate - self.min_learning_rate
            ) * (
                1 + math.cos((it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters) * math.pi)
            )

@torch.no_grad()
def gradient_clipping_(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float
):
    norm = torch.zeros(1, device=parameters[0].device)
    for p in parameters:
        if p.grad is not None:
            norm.add_(p.grad.pow(2).sum())
    norm = norm.sqrt()
    eps = 1e-6
    if norm >= max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(max_l2_norm / (norm + eps))
