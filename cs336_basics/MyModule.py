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
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = round(8 / 3 * d_model / 64) * 64
        self.d_ff = d_ff
        self.w1 = MyLinear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = MyLinear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = MyLinear(self.d_model, self.d_ff, device=device, dtype=dtype)
    
    @jaxtyped(typechecker=beartype)
    def init_with_weights(
        self,
        w1: Float[torch.Tensor, "d_ff d_model"],
        w2: Float[torch.Tensor, "d_model d_ff"],
        w3: Float[torch.Tensor, "d_ff d_model"],
    ):
        with torch.no_grad():
            self.w1.weight.copy_(w1)
            self.w2.weight.copy_(w2)
            self.w3.weight.copy_(w3)
    
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]:
        w1x = self.w1(x)
        silu = einsum(w1x, torch.sigmoid(w1x), "... d_ff, ... d_ff -> ... d_ff")
        w3x = self.w3(x) 
        return self.w2(einsum(silu, w3x, "... d_ff, ... d_ff -> ... d_ff"))

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
    in_dtype = x.dtype
    x = x.to(torch.float32)
    x_trans = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    result = x_trans / torch.sum(x_trans, dim=dim, keepdim=True)
    return result.to(in_dtype)

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

class My_multihead_self_attention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        self.w_qkv = MyLinear(d_model, num_heads * (self.d_k + self.d_k + self.d_v), device=device, dtype=dtype)
        self.w_o = MyLinear(num_heads * self.d_v, d_model, device=device, dtype=dtype)
        if theta is not None:
            self.rope = MyRoPE(theta, self.d_k, max_seq_len, device=device)
    
    @jaxtyped(typechecker=beartype)
    def init_with_weights(
        self,
        q_weight: Float[torch.Tensor, " h_d_k d_model"],
        k_weight: Float[torch.Tensor, " h_d_k d_model"],
        v_weight: Float[torch.Tensor, " h_d_v d_model"],
        o_weight: Float[torch.Tensor, " d_model h_d_v"],
    ):
        assert q_weight.shape[-2] == self.d_k * self.num_heads
        assert k_weight.shape[-2] == self.d_k * self.num_heads
        assert v_weight.shape[-2] == self.d_v * self.num_heads
        
        with torch.no_grad():
            self.w_qkv.weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=-2))
            self.w_o.weight.copy_(o_weight)
    
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "... sequence_length d_model"],
        token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
    ) -> Float[torch.Tensor, "... sequence_length d_model"]:
        Q, K, V = torch.split(
            self.w_qkv(x),
            [self.d_k * self.num_heads, self.d_k * self.num_heads, self.d_v * self.num_heads],
            dim=-1,
        )
        # d_x stands for d_k or d_v
        pat = "... (h d_x) -> h ... d_x"
        Q = rearrange(Q, pat, h=self.num_heads)
        K = rearrange(K, pat, h=self.num_heads)
        V = rearrange(V, pat, h=self.num_heads)
        seq_len = x.shape[-2]
        if hasattr(self, "rope"):
            if token_positions is None:
                pos = torch.arange(seq_len, device=x.device)
                token_positions = pos.expand(*x.shape[:-1])
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        return self.w_o(rearrange(My_Scaled_dot_product_attention(Q, K, V, mask), "h ... d_v -> ... (h d_v)"))

class MyTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.ln1 = MyRMSNorm(d_model, device=device, dtype=dtype)
        self.attn = My_multihead_self_attention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ln2 = MyRMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = MySwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "... sequence_length d_model"],
    ) -> Float[torch.Tensor, "... sequence_length d_model"]:
        y = x + self.attn(self.ln1(x))
        return y + self.ffn(self.ln2(y))

class MyTransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.token_embeddings = MyEmbedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList(
            MyTransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype) for _ in range(num_layers)
        )
        self.ln_final = MyRMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = MyLinear(d_model, vocab_size, device=device, dtype=dtype)
    
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        in_indices: Int[torch.Tensor, " batch_size sequence_length"],
    ) -> Float[torch.Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)

# Test
if __name__ == "__main__":
    device = torch.device("meta")
    model = MyTransformerLM(
        vocab_size=50257,
        context_length=1024,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=10000.0,
        dtype=torch.bfloat16
    ).to(device)
    from torchinfo import summary
    input_data = torch.ones((1, 1024), dtype=torch.long, device=device)
    summary(model, input_data=input_data)