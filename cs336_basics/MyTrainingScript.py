import torch
import MyModule
import MyLoss
import MyOptimizer
import MyData
import typer
import numpy as np

def main(
    num_iters: int = 100,
    batch_size: int = 128,
    device: str = 'cpu',
    vocab_size: int = 32000,
    context_length: int = 1024,
    d_model: int = 1600,
    num_layers: int = 48,
    num_heads: int = 25,
    d_ff: int = 6400,
    rope_theta: float = 10000.0,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8
):
    model = MyModule.MyTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    optimizer = MyOptimizer.My_AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    set_name = 'TinyStoriesV2-GPT4'
    train_set = np.load(f'data/tokenized_file/{set_name}-train.npy', mmap_mode='r')
    valid_set = np.load(f'data/tokenized_file/{set_name}-valid.npy', mmap_mode='r')
    for t in range(num_iters):
        optimizer.zero_grad()
        train_inputs, train_targets = MyData.run_get_batch(train_set, batch_size, context_length, device)
        train_outputs = model(train_inputs)
        loss = MyLoss.My_cross_entropy(train_outputs, train_targets)
        loss.backward()
        optimizer.step()
        print(f'Iteration {t}, Loss: {loss.cpu().item()}')
        if t % 10 == 0:
            model.eval()
            with torch.no_grad():
                valid_inputs, valid_targets = MyData.run_get_batch(valid_set, batch_size, context_length, device)
                valid_outputs = model(valid_inputs)
                valid_loss = MyLoss.My_cross_entropy(valid_outputs, valid_targets)
                print(f'Validation Loss: {valid_loss.cpu().item()}')
            model.train()
        
if __name__ == "__main__":
    typer.run(main)