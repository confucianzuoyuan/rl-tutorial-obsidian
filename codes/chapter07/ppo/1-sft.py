import torch
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = '../../gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 打印参数数量
print("模型参数数量：", sum(p.numel() for p in model.parameters()))

# 编码
text = "Hello, this is the first step of RLHF training."
tokens = tokenizer(text)
print(tokens)

# 解码
print(tokenizer.decode(tokens['input_ids']))

# 对一批数据进行编码
texts = ['Hello, this is the first step of RLHF training.',
         'I have a dog', 'I also have a cat']
tokens_obj = tokenizer(texts)

for tokens in tokens_obj['input_ids']:
    print(tokenizer.decode(tokens))

# 加载数据集
dataset_name = 'sst2'
ds = load_dataset(dataset_name)
print(ds)

ds_train, ds_val = ds['train'], ds['validation']
print(ds_train)

print(ds_train[6])

# A batch of rows
print(ds_train[:10])  # collation

# 对数据集进行解码


def tokenize(batch):
    return tokenizer(batch['sentence'])


map_kwargs = {
    'batched': True,
    'batch_size': 512,
    'remove_columns': ['idx', 'sentence', 'label']
}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)
tokenized_dataset_val = ds_val.map(tokenize, **map_kwargs)

print(tokenized_dataset_train[0])

print(tokenized_dataset_train[5:10])

# 从数据集解码

for i, seq in enumerate(tokenized_dataset_train[5:10]['input_ids']):
    print(f'{i+1}: {tokenizer.decode(seq)}')

# Filter out tweets shorter than 5 tokens

print(len(tokenized_dataset_train), len(tokenized_dataset_val))

tokenized_dataset_train = tokenized_dataset_train.filter(
    lambda x: len(x['input_ids']) > 5)
tokenized_dataset_val = tokenized_dataset_val.filter(
    lambda x: len(x['input_ids']) > 5)

print(len(tokenized_dataset_train), len(tokenized_dataset_val))

# 准备dataloader

# 设置PyTorch格式

tokenized_dataset_train.set_format(type='torch')
tokenized_dataset_val.set_format(type='torch')

print(tokenized_dataset_train[0])
print(tokenized_dataset_train[:5])

# Padding

# check what the pad token is set to (should be empty)
print(tokenizer.pad_token)

# check what the eos token is set to
print(tokenizer.eos_token)

# N+ Implementation paper (page 5) says otherwise
# but we would use attention_mask to remove extra eos_token used for padding
tokenizer.pad_token = tokenizer.eos_token

# Collation with Padding

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)  # labels

dataloader_params = {
    'batch_size': 32,
    'collate_fn': data_collator
}

train_dataloader = DataLoader(tokenized_dataset_train, **dataloader_params)
val_dataloader = DataLoader(tokenized_dataset_val, **dataloader_params)

print(len(train_dataloader))

batch = next(iter(train_dataloader))
print(batch.keys())

print(batch['input_ids'].shape)

print(batch['input_ids'][0])

print(batch['labels'][0])

print(batch['attention_mask'][0])

# Supervised Fine-tuning (SFT)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 1

# 训练循环


def validate(epoch):
    model.eval()
    total_loss = 0.0
    for i, batch in enumerate(val_dataloader):
        # iteration = epoch * len(val_dataloader) + i
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
            # Uses transformers.loss.loss_utils.ForCausalLMLoss for loss calculation
            loss = outputs.loss
            total_loss += loss.item()
    print(f'val_loss at {epoch} epoch:', total_loss / len(val_dataloader))


model.to(device)
validate(0)
for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        print(f'Loss: {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    validate(epoch+1)

## 保存模型

model.save_pretrained('./sft_model_epoch_1')

