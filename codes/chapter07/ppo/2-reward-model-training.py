from sklearn.metrics import confusion_matrix
from transformers import DataCollatorWithPadding, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import torch
from datasets import load_dataset
model_name = '../../gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset_name = '../../sst2'
dataset = load_dataset(dataset_name)
print(dataset)


ds_train, ds_val = dataset['train'], dataset['validation']

print(ds_train[4])

REWARD_TOKEN_ID = tokenizer.eos_token_id

print(REWARD_TOKEN_ID)


def tokenize(batch):
    outputs = tokenizer(batch['sentence'])
    outputs['score'] = [0] * len(outputs['input_ids'])
    outputs['score_index'] = [0] * len(outputs['input_ids'])
    for i in range(len(outputs['input_ids'])):
        outputs['input_ids'][i].append(REWARD_TOKEN_ID)
        outputs['attention_mask'][i].append(1)
        outputs['score'][i] = float(batch['label'][i])
        outputs['score_index'][i] = len(outputs['input_ids'][i]) - 1
    return outputs


map_kwargs = {
    "batched": True,
    "batch_size": 512,
    "remove_columns": ['idx', 'sentence', 'label']
}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)
tokenized_dataset_val = ds_val.map(tokenize, **map_kwargs)


print(tokenized_dataset_train[4])

tokenized_dataset_train.set_format(type='torch')
tokenized_dataset_val.set_format(type='torch')

print(tokenized_dataset_train[4])

# Filter out shorter tweets
tokenized_dataset_train = tokenized_dataset_train.filter(
    lambda x: len(x['input_ids']) > 6)
tokenized_dataset_val = tokenized_dataset_val.filter(
    lambda x: len(x['input_ids']) > 6)

print(len(tokenized_dataset_train))

# LLM with Reward Head


class RewardHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.reward = nn.Linear(self.hidden_size, 1)
        self._post_init()

    def _post_init(self):
        nn.init.normal_(self.reward.weight, std=(
            1.0 / np.sqrt(self.hidden_size + 1)))
        nn.init.zeros_(self.reward.bias)

    def forward(self, hidden_states):
        return self.reward(hidden_states)


class GPT2RewardHead(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_head = RewardHead(self.llm.config)

    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.llm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = transformer_outputs.hidden_states[-1]
        reward = self.reward_head(last_hidden_state).squeeze(-1)
        return torch.sigmoid(reward)


model = GPT2RewardHead(model_name)


tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorWithPadding(tokenizer)
dataloader_params = {
    'batch_size': 64,
    'shuffle': True,
    'collate_fn': data_collator
}
train_dataloader = DataLoader(tokenized_dataset_train, **dataloader_params)
val_dataloader = DataLoader(tokenized_dataset_val, **dataloader_params)

batch = next(iter(train_dataloader))
print(batch.keys())


print(batch['input_ids'][1])
print(batch['attention_mask'][1])
print(batch['score'][1])
print(batch['score_index'][1])

print(tokenizer.decode(batch['input_ids'][1]))

print(batch['attention_mask'][1].nonzero()[-1])

outputs = model(batch['input_ids'], batch['attention_mask'])

print(outputs.shape)

# Training Config

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()
num_epochs = 1  # N+ Implementation Detail paper


def validate():
    model.eval()
    total_loss = 0
    for i, batch in enumerate(val_dataloader):
        inputs = batch.to(device)
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        with torch.no_grad():
            scores = model(**model_inputs)
            batch_indices = torch.arange(scores.shape[0])
            score = scores[batch_indices, inputs['score_index']]
            target = inputs['score']
            loss = criterion(score, target)
        total_loss += loss.item()
    print('validation loss:', total_loss / len(val_dataloader))


# Training Loop

model.to(device)

validate()
for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_dataloader):
        inputs = batch.to(device)
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        scores = model(**model_inputs)
        batch_indices = torch.arange(scores.shape[0])
        score = scores[batch_indices, inputs['score_index']]
        target = inputs['score']
        loss = criterion(score, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
    validate()

torch.save(model.state_dict(), 'reward_model.pt')

validate()

# Confusion Matrix

model.eval()

all_predictions = []
all_labels = []

for i, batch in enumerate(val_dataloader):
    inputs = batch.to(device)
    model_inputs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }
    with torch.no_grad():
        scores = model(**model_inputs)
        batch_indices = torch.arange(scores.shape[0])
        score = scores[batch_indices, inputs['score_index']]
        target = inputs['score']
    predictions = (score > 0.5).int()

    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(target.cpu().numpy())

confusion_matrix(all_labels, all_predictions)
