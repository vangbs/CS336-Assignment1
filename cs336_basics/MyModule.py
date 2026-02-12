import math
import torch
from jaxtyping import jaxtyped, Float, Int, Bool
from beartype import beartype
from einops import einsum, reduce, rearrange

class MyLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std_dev = math.sqrt(2 / (in_features + out_features))
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=std_dev,
            a=-3 * std_dev,
            b=3 * std_dev,
        )
    
    @jaxtyped(typechecker=beartype)
    def init_with_weights(
        self,
        weight: Float[torch.Tensor, "out_features in_features"]
    ):
        with torch.no_grad():
            self.weight.copy_(weight)
    
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "... in_features"]
    ) -> Float[torch.Tensor, "... out_features"]:
        assert(x.shape[-1] == self.in_features)
        return einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")

class MyEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        std_dev = 1
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=std_dev,
            a=-3 * std_dev,
            b=3 * std_dev,
        )
    
    @jaxtyped(typechecker=beartype)
    def init_with_weights(
        self,
        weight: Float[torch.Tensor, "num_embeddings embedding_dim"]
    ):
        with torch.no_grad():
            self.weight.copy_(weight)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        token_ids: Int[torch.Tensor, "... sequence_length"]
    ) -> Float[torch.Tensor, "... sequence_length embedding_dim"]:
        return self.weight[token_ids]
    
class MyRMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    @jaxtyped(typechecker=beartype)
    def init_with_weights(
        self,
        weight: Float[torch.Tensor, "d_model"]
    ):
        with torch.no_grad():
            self.weight.copy_(weight)
    
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        assert(x.shape[-1] == self.d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x ** 2, "... d_model -> ... 1", "mean") + self.eps)
        result = einsum(x, self.weight, "... d_model, d_model -> ... d_model") / rms
        return result.to(in_dtype)

class MySwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = round(8 / 3 * d_model / 64) * 64
        self.w1 = MyLinear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = MyLinear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = MyLinear(self.d_model, self.d_ff, device=device, dtype=dtype)
    
    @jaxtyped(typechecker=beartype)
    def init_force_(
        self,
        d_ff: int,
        w1: Float[torch.Tensor, "d_ff d_model"],
        w2: Float[torch.Tensor, "d_model d_ff"],
        w3: Float[torch.Tensor, "d_ff d_model"],
    ):
        self.d_ff = d_ff
        assert w1.shape == (self.d_ff, self.d_model)
        self.w1 = MyLinear(self.d_model, self.d_ff, device=self.w1.weight.device, dtype=self.w1.weight.dtype)
        self.w2 = MyLinear(self.d_ff, self.d_model, device=self.w2.weight.device, dtype=self.w2.weight.dtype)
        self.w3 = MyLinear(self.d_model, self.d_ff, device=self.w3.weight.device, dtype=self.w3.weight.dtype)
        
        with torch.no_grad():
            self.w1.weight.copy_(w1)
            self.w2.weight.copy_(w2)
            self.w3.weight.copy_(w3)
    
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]:
        w1x = self.w1.forward(x)
        silu = einsum(w1x, torch.sigmoid(w1x), "... d_ff, ... d_ff -> ... d_ff")
        w3x = self.w3.forward(x) 
        return self.w2.forward(einsum(silu, w3x, "... d_ff, ... d_ff -> ... d_ff"))

class MyRoPE(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # theta[i][k] = i * 1.0/ theta ^ (2k/d), k = 0,...,d/2 - 1
        pos_indices = torch.arange(0, max_seq_len, device=device, dtype=torch.float64)
        dim_indices = torch.arange(0, d_k // 2, device=device, dtype=torch.float64)
        th = einsum(pos_indices, 1/(theta ** (2 * dim_indices / d_k)), "max_seq_len, half_d_k -> max_seq_len half_d_k")
        self.register_buffer("cos_cached", torch.cos(th), persistent=False)
        self.register_buffer("sin_cached", torch.sin(th), persistent=False)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        assert x.shape[-2] <= self.max_seq_len
        in_dtype = x.dtype
        cos_curr: Float[torch.Tensor, "... seq_len half_d_k"] = self.cos_cached[token_positions]
        sin_curr: Float[torch.Tensor, "... seq_len half_d_k"] = self.sin_cached[token_positions]
        x_even, x_odd = rearrange(
            x,
            "... seq_len (half_d_k d) -> ... seq_len half_d_k d",
            d=2
        ).unbind(dim=-1)
        cos_part = torch.stack(
            [
                einsum(x_even, cos_curr, "... seq_len half_d_k, ... seq_len half_d_k -> ... seq_len half_d_k"),
                einsum(x_odd, cos_curr, "... seq_len half_d_k, ... seq_len half_d_k -> ... seq_len half_d_k")
            ],
            dim=-1
        )
        sin_part = torch.stack(
            [
                einsum(-x_odd, sin_curr, "... seq_len half_d_k, ... seq_len half_d_k -> ... seq_len half_d_k"),
                einsum(x_even, sin_curr, "... seq_len half_d_k, ... seq_len half_d_k -> ... seq_len half_d_k")
            ],
            dim=-1
        )
        return rearrange(
            cos_part + sin_part,
            "... seq_len half_d_k d -> ... seq_len (half_d_k d)",
            d=2
        ).to(in_dtype)

def MySoftmax(
    x: Float[torch.Tensor, "..."],
    dim: int,
) -> Float[torch.Tensor, "..."]:
    x_trans = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return x_trans / torch.sum(x_trans, dim=dim, keepdim=True)

@jaxtyped(typechecker=beartype)
def My_Scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... queries d_k"],
    K: Float[torch.Tensor, "... keys d_k"],
    V: Float[torch.Tensor, "... keys d_v"],
    mask: Bool[torch.Tensor, "... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    pre_softmax = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float("-inf"))
    return einsum(MySoftmax(pre_softmax, dim=-1), V, "... queries keys, ... keys d_v -> ... queries d_v")

# Test
if __name__ == "__main__":
    model = MyRoPE(10000, 64, 6324)
    print(model.cos_cached.shape, model.sin_cached.shape)