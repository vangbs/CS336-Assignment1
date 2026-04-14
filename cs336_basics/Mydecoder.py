import numpy as np
import MyModule
import MyOptimizer
from BPE_tokenizer import BPE_tokenizer
import MyData
import torch
from config import ModelConfig, OptimizerConfig, TrainingConfig, InferenceConfig
from jaxtyping import jaxtyped, Int
from beartype import beartype
from dataclasses import replace

def load_model(
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
    valid_loss = 1.320312
    checkpoint_path = f'checkpoints/{training_config.set_name}-{training_config.batch_size}-{optimizer_config.max_learning_rate:.6f}-{valid_loss:.6f}.pt'
    MyData.load_checkpoint(
        src=checkpoint_path,
        model=model,
        optimizer=optimizer,
    )
    return model

@jaxtyped(typechecker=beartype)
def load_prompt(
    tokenizer: BPE_tokenizer,
    training_config: TrainingConfig,
) -> Int[torch.Tensor, " batch_size sequence_length"]:
    with open(f'data/prompts/prompt.txt', 'r') as inp:
        prompt_encoded = np.fromiter(
            tokenizer.encode_iterable(inp),
            dtype=np.uint16
        )
    return torch.from_numpy(prompt_encoded).to(training_config.device).long().unsqueeze(0)

def top_p_filtering(probs: torch.Tensor, top_p: float):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs >= top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    probs = probs.clone()
    probs[indices_to_remove] = 0.0
    return probs / probs.sum(dim=-1, keepdim=True)


if __name__ == "__main__":
    training_config = TrainingConfig()
    inference_config = InferenceConfig()
    vocab_path = f'data/BPE_result/{training_config.set_name}-train.pkl'
    tokenizer = BPE_tokenizer.from_files(vocab_path, ["<|endoftext|>"])
    model_config = ModelConfig()
    optimizer_config = OptimizerConfig()
    training_config = replace(
        training_config,
        batch_size = 32
    )
    optimizer_config = replace(
        optimizer_config,
        max_learning_rate = 0.001000
    )
    model = load_model(model_config, optimizer_config, training_config)
    model.eval()
    prompt_encoded = load_prompt(tokenizer, training_config)
    
    with torch.no_grad():
        for t in range(inference_config.max_new_tokens):
            logits = model(prompt_encoded)[:, -1, :] # Shape: (batch_size, vocab_size)
            logits /= inference_config.temperature
            probs = MyModule.MySoftmax(logits, dim=-1)
            probs = top_p_filtering(probs, inference_config.top_p)
            
            next_token = torch.multinomial(probs, num_samples=1)
            prompt_encoded = torch.cat([prompt_encoded, next_token], dim=-1)
            # Batchsize must be 1
            if next_token.item() == inference_config.eot_index:
                break
    
    
    result_list = prompt_encoded.squeeze(0).cpu().tolist()
    print(tokenizer.decode(result_list))
    
