import torch
import MyModule
import MyLoss
import MyOptimizer
import MyData
import numpy as np
from config import ModelConfig, OptimizerConfig, TrainingConfig
import time
from dataclasses import replace
import itertools


def main(
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    training_config: TrainingConfig,
):
    model = MyModule.MyTransformerLM(
        vocab_size=model_config.vocab_size,
        context_length=model_config.context_length,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        rope_theta=model_config.rope_theta,
        device=training_config.device,
        dtype=training_config.dtype
    )
    
    optimizer = MyOptimizer.MyAdamW(
        model.parameters(),
        lr=optimizer_config.lr,
        weight_decay=optimizer_config.weight_decay,
        betas=optimizer_config.betas,
        eps=optimizer_config.eps,
    )
    scheduler = MyOptimizer.My_Cosine_Scheduler(
        optimizer,
        max_learning_rate=optimizer_config.max_learning_rate,
        min_learning_rate=optimizer_config.min_learning_rate,
        warmup_iters=optimizer_config.warmup_iters,
        cosine_cycle_iters=training_config.num_iters,
    )
    train_set = np.load(f'data/tokenized_file/{training_config.set_name}-train.npy', mmap_mode='r')
    valid_set = np.load(f'data/tokenized_file/{training_config.set_name}-valid.npy', mmap_mode='r')
    start_time = time.time()
    with open(f'logs/log_{training_config.set_name}_{training_config.batch_size * training_config.accumulation_steps}_{optimizer_config.max_learning_rate:.6f}.txt', 'a') as log_file:
        for t in range(1, training_config.num_iters + 1):
            optimizer.zero_grad()
            for _ in range(training_config.accumulation_steps):
                train_inputs, train_targets = MyData.run_get_batch(train_set, training_config.batch_size, model_config.context_length, training_config.device)
                train_outputs = model(train_inputs)
                loss = MyLoss.My_cross_entropy(train_outputs, train_targets)
                loss = loss / training_config.accumulation_steps
                loss.backward()
            
            MyOptimizer.gradient_clipping_(model.parameters(), training_config.max_grad_norm)
            scheduler.step(t)
            optimizer.step()
            if t % 250 == 0:
                print(f'Iteration {t}, Loss: {loss.cpu().item()}, time: {(time.time() - start_time) / 60} minutes', file=log_file, flush=True)
                model.eval()
                with torch.no_grad():
                    valid_inputs, valid_targets = MyData.run_get_batch(valid_set, training_config.batch_size, model_config.context_length, training_config.device)
                    valid_outputs = model(valid_inputs)
                    valid_loss = MyLoss.My_cross_entropy(valid_outputs, valid_targets).cpu().item()
                    print(f'Validation Loss: {valid_loss}', file=log_file, flush=True)
                model.train()
    
    MyData.save_checkpoint(
        model,
        optimizer,
        training_config.num_iters,
        f'checkpoints/{training_config.set_name}-{training_config.batch_size * training_config.accumulation_steps}-{optimizer_config.max_learning_rate:.6f}.pt'
    )
    
        
if __name__ == "__main__":
    base_model_cfg = ModelConfig()
    base_opt_cfg = OptimizerConfig()
    base_train_cfg = TrainingConfig()


    search_space = {
        "batch_size": [32],
        "accumulation_steps": [4, 8, 16],
        "max_learning_rate": [1e-3, 5e-3, 1e-2]
    }

    batch_size_times_num_iters = 1280000
    
    keys, values = zip(*search_space.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        print(params)
        training_config = replace(
            base_train_cfg,
            batch_size = params["batch_size"],
            accumulation_steps = params["accumulation_steps"],
            num_iters = batch_size_times_num_iters // params["batch_size"] // params["accumulation_steps"]
        )
        optimizer_config = replace(
            base_opt_cfg,
            max_learning_rate = params["max_learning_rate"],
            min_learning_rate = params["max_learning_rate"] / 10,
            warmup_iters = training_config.num_iters // 10
        )
        main(base_model_cfg, optimizer_config, training_config)