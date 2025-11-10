import matplotlib.pyplot as plt
import torch

from qwen3 import Qwen3Model, Qwen3Tokenizer, generate_text_simple
from data import create_dataloader_v1

# 将文本转换成input_ids
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加批次的维度
    return encoded_tensor

# 将input_ids转换成文本
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 删除批次的维度
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算一个批次的损失"""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 输入的批次数据的预测出的数据
    logits = model(input_batch)
    # 预测的logits和目标值计算交叉熵
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.cfg["context_length"]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=500, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # 主训练循环
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def main(qwen3_config, settings):
    torch.manual_seed(123)
    device = torch.device("cuda")
    file_path = "the-verdict.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    model = Qwen3Model(qwen3_config) # 随机权重
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=qwen3_config["context_length"],
        stride=qwen3_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=qwen3_config["context_length"],
        stride=qwen3_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    tokenizer_file_path = "tokenizer.json"

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path, apply_chat_template=False
    )

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=settings["num_epochs"],
        eval_freq=5,
        eval_iter=1,
        start_context="I like to fancy that",
        tokenizer=tokenizer,
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
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

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 4,
        "weight_decay": 0.1,
    }

    train_losses, val_losses, tokens_seen, model = main(QWEN3_CONFIG, OTHER_SETTINGS)

    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    torch.save(model.state_dict(), "model.pth")
    model = Qwen3Model(QWEN3_CONFIG)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
