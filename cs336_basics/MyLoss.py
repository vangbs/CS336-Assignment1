import torch
from jaxtyping import jaxtyped, Float, Int
from beartype import beartype
from einops import rearrange

@jaxtyped(typechecker=beartype)
def My_cross_entropy(
    inputs: Float[torch.Tensor, "*batch vocab_size"],
    targets: Int[torch.Tensor, "*batch"]
) -> Float[torch.Tensor, ""]:
    x = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    y = rearrange(
        x.gather(
            dim=-1,
            index = rearrange(targets, "... -> ... 1")
        ),
        "... 1 -> ..."
    )
    return -torch.mean(y - torch.log(torch.sum(torch.exp(x), dim=-1)))

if __name__ == "__main__":
    inputs = torch.tensor(
        [
            [
                [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
            ],
            [
                [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
            ],
        ]
    )
    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    print(My_cross_entropy(inputs, targets).shape)
