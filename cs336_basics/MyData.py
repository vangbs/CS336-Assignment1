import torch
import numpy as np
import numpy.typing as npt
import os
import typing
import cs336_basics.MyOptimizer as MyOptimizer

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
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    scheduler: MyOptimizer.My_Cosine_Scheduler | None = None
):
    torch.save((
        model.state_dict(),
        optimizer.state_dict(),
        iteration,
        scheduler.state_dict() if scheduler is not None else None
    ) ,out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: MyOptimizer.My_Cosine_Scheduler | None = None
) -> int:
    model_state, optimizer_state, iteration, scheduler_state = torch.load(src)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
        scheduler.optimizer = optimizer
    return iteration


if __name__ == "__main__":
    run_get_batch(
        dataset=np.arange(0, 100),
        batch_size=5,
        context_length=3,
        device="cpu"
    )
