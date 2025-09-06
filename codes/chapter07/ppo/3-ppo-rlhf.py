# Load the SFT and Reward Models

# Reward model

from transformers import DataCollatorWithPadding
from copy import deepcopy
from torch.utils.data import DataLoader
import random
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from typing import Optional
from torch import nn
import numpy as np
from transformers import AutoModelForCausalLM


class RewardHead(nn.Module):
    """
    The RewardHead class implements a head for GPT2
    that returns a scalar for each output token.
    """

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
        output = hidden_states
        return self.reward(output)


class GPT2RewardModel(nn.Module):
    """
    GPT2 model with a reward head on top.
    """

    def __init__(self, model_name):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        # config = self.llm.config
        # Add the reward head
        self.reward_head = RewardHead(self.llm.config)

    def forward(
        self,
        input_ids,
        attention_mask,
    ) -> Optional[torch.FloatTensor]:

        transformer_outputs = self.llm.forward(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get the last hidden state
        last_hidden_state = transformer_outputs.hidden_states[-1]

        # Apply the reward head
        rewards = self.reward_head(last_hidden_state).squeeze(-1)

        return torch.sigmoid(rewards)


model_name = "../../gpt2"
reward_model = GPT2RewardModel(model_name)
reward_model.load_state_dict(torch.load("reward_model.pt", map_location='cpu'))

# Model with Value Head


class ValueHead(nn.Module):
    """
    The ValueHead class implements a head for GPT2
    that returns a scalar for each output token.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.value = nn.Linear(self.hidden_size, 1)
        self._post_init()

    def _post_init(self):
        nn.init.normal_(self.value.weight, std=(
            1.0 / np.sqrt(self.hidden_size + 1)))
        nn.init.zeros_(self.value.bias)

    def forward(self, hidden_states):
        output = hidden_states
        return self.value(output)


class ModelForCausalLMWithValueHead(nn.Module):
    """
    GPT2 model with a value head on top.
    """

    def __init__(self, model_path):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_path)
        # config = self.llm.config
        # Add the reward head
        self.v_head = ValueHead(self.llm.config)

    def forward(
        self,
        input_ids,
        attention_mask,
    ) -> Optional[torch.FloatTensor]:

        transformer_outputs = self.llm.forward(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        lm_logits = transformer_outputs.logits
        # Get the last hidden state
        last_hidden_state = transformer_outputs.hidden_states[-1]

        # Apply the reward head
        value = self.v_head(last_hidden_state).squeeze(-1)
        return lm_logits, value

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)


model_path = './sft_model_epoch_1'
model = ModelForCausalLMWithValueHead(model_path)

# Preparing Dataset

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("../../sst2")
print(dataset)


ds_train, ds_val = dataset['train'], dataset['validation']

# Filtering

print(len(ds_train))

ds_train = ds_train.filter(lambda x: len(x['sentence'].split(' ')) > 8)

print(len(ds_train))

ds_val = ds_val.filter(lambda x: len(x['sentence'].split(' ')) > 8)

print(len(ds_val))

input_min_token_length = 2
input_max_token_length = 8
input_token_length_range = list(
    range(input_min_token_length, input_max_token_length))
print(input_token_length_range)

print(random.choice(input_token_length_range))


def tokenize(sample):
    input_size = random.choice(input_token_length_range)
    sample['input_ids'] = tokenizer.encode(sample['sentence'])[:input_size]
    sample['attention_mask'] = [1] * len(sample['input_ids'])
    sample['query'] = tokenizer.decode(sample['input_ids'])
    return sample


map_kwargs = {
    "batched": False,
    "remove_columns": ['idx', 'sentence', 'label']
}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)
tokenized_dataset_val = ds_val.map(tokenize, **map_kwargs)

tokenized_dataset_train.set_format(type='torch')
tokenized_dataset_val.set_format(type='torch')

print(tokenized_dataset_train[6])

REWARD_TOKEN_ID = tokenizer.eos_token_id


batch_size = 32


def collator(batch):
    return dict((key, [d[key] for d in batch]) for key in batch[0])


train_dataloader = DataLoader(
    tokenized_dataset_train, batch_size=batch_size, collate_fn=collator, shuffle=True)
val_dataloader = DataLoader(
    tokenized_dataset_val, batch_size=batch_size, collate_fn=collator, shuffle=True)

batch = next(iter(train_dataloader))
print(batch)

output_min_length = 5
output_max_length = 16

# https://huggingface.co/docs/trl/how_to_train#how-to-generate-text-for-training

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id
}

# Sample Generation

new_tokens = random.choice(list(range(output_min_length, output_max_length)))
generation_kwargs["max_new_tokens"] = new_tokens
sample = tokenizer('Hi, this')
print(sample)

query_response = model.generate(
    input_ids=torch.tensor(sample['input_ids']).unsqueeze(0),
    attention_mask=torch.tensor(sample['attention_mask']).unsqueeze(0),
    **generation_kwargs
).squeeze(0)
print(query_response)


print(tokenizer.decode(query_response))

with torch.no_grad():
    query_response_score = torch.cat(
        [query_response, torch.tensor([REWARD_TOKEN_ID])])
    attention_mask = torch.ones_like(query_response_score, dtype=torch.long)
    score = reward_model(query_response_score.unsqueeze(
        0), attention_mask.unsqueeze(0)).squeeze(0)[-1]
print(score)

# Batch Generation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
reward_model = reward_model.to(device)

query_tensors = batch['input_ids']
query_attention_masks = batch['attention_mask']

response_tensors = []
query_response_tensors = []
score_tensors = []

for i, query in enumerate(query_tensors):
    query = query.to(device)
    query_attention_mask = query_attention_masks[i].to(device)
    new_tokens = random.choice(
        list(range(output_min_length, output_max_length)))
    generation_kwargs["max_new_tokens"] = new_tokens
    query_response = model.generate(
        input_ids=query.unsqueeze(0),
        attention_mask=query_attention_mask.unsqueeze(0),
        **generation_kwargs
    ).squeeze(0)

    response_len = len(query_response) - len(query)
    response_tensors.append(query_response[-response_len:])
    query_response_tensors.append(query_response)

    with torch.no_grad():
        query_response_score = torch.cat(
            [query_response, torch.tensor([REWARD_TOKEN_ID]).to(device)])
        attention_mask = torch.ones_like(
            query_response_score, dtype=torch.long)
        score = reward_model(query_response_score.unsqueeze(
            0), attention_mask.unsqueeze(0)).squeeze(0)[-1]
        score = 2 * (score - 0.5)
    score_tensors.append(score)

batch["response"] = [tokenizer.decode(response)
                     for response in response_tensors]
print(batch['response'])

# Compute Reward

# reward = score - \log(\pi_\theta^{RL}/\pi^{SFT})

sft_model = deepcopy(model)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

input_data = data_collator([
    {'input_ids': ids,
     'attention_mask': torch.ones_like(ids)} for ids in query_response_tensors
]).to(device)
print(input_data)


def compute_rewards(input_data, query_tensors, response_tensors, score_tensors):
    with torch.no_grad():
        logits, values = model(**input_data)  # b, seq, vocab
        ref_logits, _ = sft_model(**input_data)
        logp = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        ref_logp = torch.nn.functional.log_softmax(
            ref_logits[:, :-1, :], dim=-1)

        labels = input_data['input_ids'][:, 1:]  # b, seq

        logp = torch.gather(logp, 2, labels.unsqueeze(-1)
                            ).squeeze(-1)  # batch, seq
        ref_logp = torch.gather(
            ref_logp, 2, labels.unsqueeze(-1)).squeeze(-1)  # batch, seq

        kl = logp - ref_logp
        beta = 0.2
        rewards = - beta * kl
        attention_mask = input_data['attention_mask']
        masks = torch.zeros_like(attention_mask[:, 1:])
        masks[:, :] = attention_mask[:, 1:]
        for j in range(len(query_tensors)):
            start = len(query_tensors[j]) - 1
            end = start + len(response_tensors[j])
            masks[j, :start] = 0
            masks[j, end:] = 0
            rewards[j, end - 1] += score_tensors[j]
            rewards[j, :] *= masks[j, :]
            values[j, :-1] *= masks[j, :]

    return logp, rewards, values[:, :-1], masks


logprobs, rewards, values, masks = compute_rewards(
    input_data, query_tensors, response_tensors, score_tensors)
print(rewards[0])
print(input_data['input_ids'][0])
print(input_data['attention_mask'][0])

print(masks[0])
print(values[0])

# Compute Advantage


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum()


def masked_var(values, mask):
    mean = masked_mean(values, mask)
    centred_values = values - mean
    return masked_mean(centred_values ** 2, mask)


def masked_whiten(values, mask):
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    whitened += mean
    return whitened


def compute_advantage(rewards, values, masks):
    lastgae = 0.0
    advantage_reversed = []
    seq_length = rewards.shape[-1]
    gamma, lam = 1.0, 0.95

    for t in reversed(range(seq_length)):
        nextvalues = values[:, t + 1] if t < seq_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgae = delta + gamma * lam * lastgae
        advantage_reversed.append(lastgae)
    advantages = torch.stack(advantage_reversed[::-1], dim=1)
    advantages = masked_whiten(advantages, masks)

    returns = advantages + values
    return advantages, returns


advantages, returns = compute_advantage(rewards, values, masks)
print(advantages[0])
print(returns[0])


# Mini-batch PPO Training

# Training Config

learning_rate = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(np.random.permutation(batch_size))

mini_batch_size = 4
ppo_epochs = 4

cliprange_ratio = 0.2

v_loss_coeff = 0.1

ratio_threshold = 10


def compute_loss(old_logprobs, values, logprobs, vpreds, masks, advantages, returns):
    ratio = torch.exp(logprobs - old_logprobs)
    pg_loss1 = - ratio * advantages
    pg_loss2 = - torch.clamp(ratio, 1 - cliprange_ratio,
                             1 + cliprange_ratio) * advantages
    pg_loss = masked_mean(torch.max(pg_loss1, pg_loss2), masks)

    v_loss = masked_mean((vpreds - returns) ** 2, masks)
    loss = pg_loss + v_loss_coeff * v_loss

    avg_ratio = masked_mean(ratio, masks)
    if avg_ratio > ratio_threshold:
        pg_loss = pg_loss * 0.0
        v_loss = v_loss * 0.0
        loss = loss * 0.0

    return loss, v_loss


def mini_batch_train():
    for ep in range(ppo_epochs):
        batch_inds = np.random.permutation(batch_size)

        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mini_batch_inds = batch_inds[start:end]

            mb_model_inputs = {
                'input_ids': input_data['input_ids'][mini_batch_inds],
                'attention_mask': input_data['attention_mask'][mini_batch_inds]
            }
            mb_logits, mb_vpreds = model(**mb_model_inputs)
            mb_logits = torch.nn.functional.log_softmax(
                mb_logits[:, :-1, :], dim=-1)
            mb_logprobs = torch.gather(
                mb_logits, 2, mb_model_inputs['input_ids'][:, 1:].unsqueeze(-1)).squeeze(-1)

            loss, loss_v = compute_loss(
                logprobs[mini_batch_inds],
                values[mini_batch_inds],
                mb_logprobs,
                mb_vpreds[:, :-1],
                masks[mini_batch_inds],
                advantages[mini_batch_inds],
                returns[mini_batch_inds]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss/total', loss.item())
    print('mini-batch training finished')


mini_batch_train()

# Train RLHF

num_epochs = 1

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Generate responses
        query_tensors = batch['input_ids']
        query_attention_masks = batch['attention_mask']

        response_tensors = []
        query_response_tensors = []
        score_tensors = []

        for i, query in enumerate(query_tensors):
            query = query.to(device)
            query_attention_mask = query_attention_masks[i].to(device)
            new_tokens = random.choice(
                list(range(output_min_length, output_max_length)))
            generation_kwargs["max_new_tokens"] = new_tokens
            query_response = model.generate(
                input_ids=query.unsqueeze(0),
                attention_mask=query_attention_mask.unsqueeze(0),
                **generation_kwargs
            ).squeeze(0)

            response_len = len(query_response) - len(query)
            response_tensors.append(query_response[-response_len:])
            query_response_tensors.append(query_response)

            with torch.no_grad():
                query_response_score = torch.cat(
                    [query_response, torch.tensor([REWARD_TOKEN_ID]).to(device)])
                attention_mask = torch.ones_like(
                    query_response_score, dtype=torch.long)
                score = reward_model(query_response_score.unsqueeze(
                    0), attention_mask.unsqueeze(0)).squeeze(0)[-1]
                score = 2 * (score - 0.5)
            score_tensors.append(score)

        input_data = data_collator([
            {
                'input_ids': ids,
                'attention_mask': torch.ones_like(ids)
            }
            for ids in query_response_tensors
        ]).to(device)

        # rewards and advantages
        logprobs, rewards, values, masks = compute_rewards(
            input_data, query_tensors, response_tensors, score_tensors)
        advantages, returns = compute_advantage(rewards, values, masks)

        # mini batch training
        mini_batch_train()
    print(f'epoch {epoch + 1} finished')

# Validation

print(len(tokenized_dataset_val))

val_gen_lengths = [0] * len(tokenized_dataset_val)
for i in range(len(tokenized_dataset_val)):
    val_gen_lengths[i] = random.choice(
        list(range(output_min_length, output_max_length)))

print(val_gen_lengths[:10])


def validate():
    scores = []
    for b, batch in enumerate(val_dataloader):
        # Generate_responses
        query_tensors = batch['input_ids']
        query_attention_masks = batch['attention_mask']
        for i, query in enumerate(query_tensors):
            query = query.to(device)
            query_attention_mask = query_attention_masks[i].to(device)
            new_tokens = val_gen_lengths[b * len(query_tensors) + i]
            generation_kwargs["max_new_tokens"] = new_tokens
            query_response = model.generate(
                input_ids=query.unsqueeze(0),
                attention_mask=query_attention_mask.unsqueeze(0),
                **generation_kwargs
            ).squeeze(0)
            query_response_score = torch.cat(
                [query_response, torch.tensor([REWARD_TOKEN_ID]).to(device)])
            attention_mask = torch.ones_like(
                query_response_score, dtype=torch.long)
            score = reward_model(query_response_score.unsqueeze(
                0), attention_mask.unsqueeze(0)).squeeze(0)[-1]
            score = 2 * (score - 0.5)
            scores.append(score.item())
    print('avg score:', sum(scores) / len(scores))


validate()

torch.save(model.state_dict(), 'ppo_model_epoch_1.pt')

model_path = './sft_model_epoch_1'
model = ModelForCausalLMWithValueHead(model_path).to(device)

validate()
