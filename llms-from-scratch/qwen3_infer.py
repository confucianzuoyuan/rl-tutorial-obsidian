import torch
from qwen3 import Qwen3Model, Qwen3Tokenizer
from qwen3_train import generate_and_print_sample

if __name__ == "__main__":
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    QWEN3_CONFIG = {
        "vocab_size": 151_936,  # Vocabulary size
        "context_length": 256,  # Context length that was used to train the model
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 16,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "hidden_dim": 3072,  # Size of the intermediate dimension in FeedForward
        "head_dim": 128,  # Size of the heads in GQA
        "qk_norm": True,  # Whether to normalize queries and keys in GQA
        "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
        "dtype": torch.bfloat16,  # Lower-precision dtype to reduce memory usage
    }

    model = Qwen3Model(QWEN3_CONFIG)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.to(device)

    tokenizer_file_path = "Qwen3-0.6B-Base/tokenizer.json"

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path, apply_chat_template=False
    )
    
    generate_and_print_sample(model, tokenizer, device, "巴蜀，历来有天府之国的美誉，")
