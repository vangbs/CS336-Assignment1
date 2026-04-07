from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 10000,
    context_length: int = 256,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    d_ff: int = 1344,
    rope_theta: float = 10000.0,

@dataclass(frozen=True)
class OptimizerConfig:
    max_learning_rate: float = 6e-4,
    min_learning_rate: float = 6e-5,
    warmup_iters: int = 2000,
    cosine_cycle_iters: int = 100000,    
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-5,

@dataclass(frozen=True)
class TrainingConfig:
    num_iters: int = 100,
    batch_size: int = 128,
    max_grad_norm: float = 1.0,
    device: str = 'cpu',
    dtype: torch.dtype = torch.bfloat16,
    set_name: str = 'TinyStoriesV2-GPT4',

@dataclass(frozen=True)
class InferenceConfig:
    max_new_tokens: int = 100,
    temperature: float = 1,
    top_p: float = 1,
    eot_index: int = 0,

