import torch
import numpy as np
import numpy.typing as npt
import os
import typing

def run_get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    start_index = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    inputs = [torch.from_numpy(dataset[index: index + context_length]) for index in start_index]
    targets = [torch.from_numpy(dataset[index + 1: index + context_length + 1]) for index in start_index]
    return (
        torch.stack(inputs).to(device).long(),
        torch.stack(targets).to(device).long(),
    )

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    torch.save((model.state_dict(), optimizer.state_dict(), iteration) ,out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    model_state, optimizer_state, iteration = torch.load(src)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return iteration


if __name__ == "__main__":
    run_get_batch(
        dataset=np.arange(0, 100),
        batch_size=5,
        context_length=3,
        device="cpu"
    )
