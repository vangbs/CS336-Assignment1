import torch
import math
from collections.abc import Iterable

class MyAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float
    ):
        decay_params = [p for p in params if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in params if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        super().__init__(optim_groups, defaults={
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        })
    
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
                if "master_p" not in state:
                    state["master_p"] = p.detach().clone().to(torch.float32)
                master_p = state["master_p"]
                grad = p.grad.to(torch.float32)
                m = state.get("m", torch.zeros_like(p, dtype=torch.float32))
                v = state.get("v", torch.zeros_like(p, dtype=torch.float32))
                t = state.get("t", 0)
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * (grad ** 2)
                t += 1
                alpha_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                
                master_p.sub_(m / (torch.sqrt(v) + eps), alpha=alpha_t)
                master_p.mul_(1 - lr * weight_decay)
                p.copy_(master_p)
                
                state["m"] = m
                state["v"] = v
                state["t"] = t
                state["master_p"] = master_p

        return None

class My_Cosine_Scheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ):
        self.optimizer = optimizer
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def state_dict(self):
        return {
            "max_learning_rate": self.max_learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "warmup_iters": self.warmup_iters,
            "cosine_cycle_iters": self.cosine_cycle_iters,
        }
    
    def load_state_dict(self, state_dict):
        self.max_learning_rate = state_dict["max_learning_rate"]
        self.min_learning_rate = state_dict["min_learning_rate"]
        self.warmup_iters = state_dict["warmup_iters"]
        self.cosine_cycle_iters = state_dict["cosine_cycle_iters"]
    
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
    
    def step(self, it: int):
        lr = self.get_learning_rate(it)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        

@torch.no_grad()
def gradient_clipping_(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float
):
    norm = torch.zeros(1, device=parameters[0].device, dtype=torch.float32)
    for p in parameters:
        if p.grad is not None:
            norm.add_(p.grad.to(torch.float32).pow(2).sum())
    norm = norm.sqrt()
    eps = 1e-6
    if norm >= max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(max_l2_norm / (norm + eps))
