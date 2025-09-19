## 1. 预训练大语言模型

载入 gpt2 的 124M 参数的模型。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = './gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# 打印模型的参数数量
print(sum(p.numel() for p in model.parameters()))
```

测试一下分词器Tokenizer

Encoding编码功能

```python
text = "Hello, this is the first step of RLHF training."
tokens = tokenizer(text)
print(tokens)
```

Decoding解码功能

```python
print(tokenizer.decode(tokens['input_ids']))
```

对一批数据进行编码

```python
texts = [
    'Hello, this is the first step of RLHF training.',
    'I have a dog',
    'I also have a cat'
]
tokens_obj = tokenizer(texts)

for tokens in tokens_obj['input_ids']:
    print(tokenizer.decode(tokens))
```

看看gpt2预训练大模型的输出

```python
from transformers import pipeline, set_seed
from pprint import pprint
# 生成式任务
g = pipeline('text-generation', model='./gpt2')
set_seed(42)
# 给定提示词，输出完整内容
pprint(g(
    "this is a",
    max_length=30,
    num_return_sequences=1
))
```

可以看到下面的输出，是基本上没有什么意义的。这也符合预训练大模型的特点。是一块没有被打磨过的璞玉。

```bash
[{'generated_text': 'this is a very good question that everyone can answer, '
                    'and I have no doubt that it is a common misconception '
                    'that there are no women in this country, and this is a '
                    'very difficult question to answer. But it is indeed true '
                    'that there are many women here who have had their fathers '
                    'killed or shot, and that is something that we all must '
                    'take into account and address before we talk about the '
                    'statistics that we have."\n'
                    '\n'
                    'It is not the first time that a man has been killed or '
                    'shot by a police officer. In 1986, a man was shot in the '
                    'head by a cop at the hands of a suspect in a shooting at '
                    "a McDonald's. It was claimed by the San Francisco cop "
                    'that he was trying to stop the suspect and that the '
                    'suspect shot the suspect in self-defense. The police '
                    'officer was fired, and the man was shot in the head.\n'
                    '\n'
                    'If there were no reports of shootings, it would be '
                    'impossible to know why anyone would go to the police '
                    'station and do something like that and not do it. Even '
                    'though there are cases where police officers are falsely '
                    'accused, they are not prosecuted. That is why officers '
                    'are usually not indicted for their actions. If the police '
                    'officer is charged with a crime, they have to plead '
                    'guilty to a crime and then face'}]
```

接下来我们需要准备一份数据集来对这个预训练大模型进行微调，对它打磨一下。

## 2. 监督微调（SFT）

我们使用sst2数据集。这是一份电影评论的数据集。质量很高。

![[ultrachat数据集.png]]

标签是评论的情感。我们现在只使用 **文本部分** sentence字段来对 gpt2 进行监督微调。

导入数据集

```python
from datasets import load_dataset
dataset_path = './sst2'
ds = load_dataset(dataset_path)
ds_train, ds_val = ds['train'], ds['validation']

print(ds)
print(ds_train)
print(ds_train[6])
print(ds_train[:10])
```

对数据集进行分词

```python
# 只使用文本内容sentence，不使用情感标签
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
```

对数据集进行解码

```python
for i, seq in enumerate(tokenized_dataset_train[5:10]['input_ids']):
    print(f'{i+1}: {tokenizer.decode(seq)}')
```

去掉少于 6 个 token 的文本

```python
print(len(tokenized_dataset_train), len(tokenized_dataset_val))

tokenized_dataset_train = tokenized_dataset_train.filter(
    lambda x: len(x['input_ids']) > 5)
tokenized_dataset_val = tokenized_dataset_val.filter(
    lambda x: len(x['input_ids']) > 5)

print(len(tokenized_dataset_train), len(tokenized_dataset_val))
```

准备 dataloader 数据加载器

设置为 PyTorch 的数据格式

```python
tokenized_dataset_train.set_format(type='torch')
tokenized_dataset_val.set_format(type='torch')

print(tokenized_dataset_train[0])
print(tokenized_dataset_train[:5])
```

填充 Padding

```python
# 检查pad token的设置（应该为空）
print(tokenizer.pad_token)
# # 检查eos token的设置
print(tokenizer.eos_token)
# N+ Implementation论文（第5页）说法不同
# 但我们会使用attention_mask来移除用于填充的额外eos_token
# 通过attention_mask来区分真正的结束token和用于填充的token
tokenizer.pad_token = tokenizer.eos_token
```

为什么这么设置？

```python
# 示例说明
text1 = "Hello world"           # 短文本
text2 = "Hello world how are you today"  # 长文本

# 填充后：
# text1: "Hello world<|endoftext|><|endoftext|><|endoftext|>"  # 后面的是填充
# text2: "Hello world how are you today<|endoftext|>"         # 最后的是真正结束

# attention_mask区分：
# text1: [1, 1, 1, 0, 0, 0]  # 1表示真实token，0表示填充token
# text2: [1, 1, 1, 1, 1, 1, 1, 1]  # 全部都是真实token
```

使用 Padding 整理（collation）数据

```python
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
# mlm=False，将数据整理成“因果语言建模”需要的数据格式
# “因果语言建模”就是“预测下一个token”类型的任务，也就是gpt风格的自回归模型
# 如果mlm=True，那么数据整理成bert风格的任务所需的数据格式
data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False
) # labels

dataloader_params = {
    'batch_size': 16, # 6G显存正好够用
    'collate_fn': data_collator
}

train_dataloader = DataLoader(
    tokenized_dataset_train,
    **dataloader_params
)
val_dataloader = DataLoader(
    tokenized_dataset_val,
    **dataloader_params
)

print(len(train_dataloader))

batch = next(iter(train_dataloader))
print(batch.keys())
print(batch['input_ids'].shape)
print(batch['input_ids'][0])
print(batch['labels'][0])
print(batch['attention_mask'][0])
```

监督微调

```python
import torch
# 要更新的是model的参数
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# 一般sft会训练1个epoch，也就是把训练数据看一遍就可以了
# 否则容易过拟合，造成“灾难性遗忘”
num_epochs = 1
```

训练循环

```python
def validate(epoch):
    """验证函数"""
    model.eval() # 禁用模型的随机性，例如dropout等特性
    total_loss = 0.0
    for i, batch in enumerate(val_dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss # 损失
            total_loss += loss.item()
    print(f'val_loss at {epoch} epoch:', total_loss / len(val_dataloader))
```

训练

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
```

保存模型

```python
model.save_pretrained('./gpt2-sft')
tokenizer.save_pretrained('./gpt2-sft')
```

测试一下输出

```python
from transformers import pipeline, set_seed
from pprint import pprint
g = pipeline('text-generation', model='./gpt2-sft')
set_seed(42)
pprint(g("this is a", max_length=30, num_return_sequences=1))
```

输出结果是

```bash
[{'generated_text': 'this is a movie you want to '
                    'watch'}]
```

可以看到，经过了sft，模型不再乱输出了，而是输出了电影相关的描述。

## 3. 奖励模型（Reward Model）的训练

之前在使用PPO玩倒立摆游戏时，我们是不需要奖励模型的，因为没有木杆只要不倒下，倒立摆环境就会给出即时奖励 1 。

而在大语言模型这个环境中，我们如何判断上面sft微调过的大模型的输出是否符合人类的偏好呢？

我们想让上面的sft微调后的模型输出 **正向情感** 的电影评论。

首先就需要训练一个奖励模型，然后奖励模型会对大模型的输出给出评分。然后更新大模型的权重。

我们的思路还是将一个124M参数的gpt2预训练模型微调成为一个 **奖励模型** 。

```python
from transformers import AutoTokenizer
model_name = './gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

导入数据集

```python
from datasets import load_dataset
dataset_path = './sst2'
dataset = load_dataset(dataset_path)
print(dataset)

ds_train, ds_val = dataset['train'], dataset['validation']
print(ds_train[4])
```

对数据集进行分词处理

```python
# 定义一个新的token，奖励token，eos
REWARD_TOKEN_ID = tokenizer.eos_token_id
print(REWARD_TOKEN_ID)

def tokenize(batch):
    # 提取出文本内容
    outputs = tokenizer(batch['sentence'])
    # 每条数据一个评分，初始化为 0 。
    outputs['score'] = [0] * len(outputs['input_ids'])
    # 对每条数据的最后的reward token进行评分
    outputs['score_index'] = [0] * \
                             len(outputs['input_ids'])
    for i in range(len(outputs['input_ids'])):
        # 第 i 条数据的末尾添加一个 eos token，作为reward token
        outputs['input_ids'][i].append(REWARD_TOKEN_ID)
        # reward token的掩码设置为 1 。
        outputs['attention_mask'][i].append(1)
        # 正向情感的文本评分为 1 。负向情感的评分为 0 。
        outputs['score'][i] = float(batch['label'][i])
        # 对 reward token 进行评分，也就是评分的索引为 reward token 的索引。
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
```

设置 PyTorch 数据格式

```python
tokenized_dataset_train.set_format(type='torch')
tokenized_dataset_val.set_format(type='torch')

print(tokenized_dataset_train[4])
```

将 token 数量小于 7 的文本过滤掉

```python
tokenized_dataset_train = tokenized_dataset_train.filter(
    lambda x: len(x['input_ids']) > 6)
tokenized_dataset_val = tokenized_dataset_val.filter(
    lambda x: len(x['input_ids']) > 6)

print(len(tokenized_dataset_train))
```

奖励模型的定义：gpt2 + 线性层

![[reward-model.excalidraw|800]]

```python
import torch
from torch import nn
import numpy as np
from transformers import AutoModelForCausalLM

class RewardHead(nn.Module):
    """奖励“头”，是一个线性层"""
    def __init__(self, config):
        super().__init__()
        # gpt2最后输出的隐藏层的维度
        self.hidden_size = config.hidden_size
        # 线性层用来对gpt2最后输出的隐藏层给奖励
        self.reward = nn.Linear(self.hidden_size, 1)
        self._post_init()

    def _post_init(self):
        # 使用正态分布初始化权重
        nn.init.normal_(
            self.reward.weight,
            std=(1.0 / np.sqrt(self.hidden_size + 1))
        )
        # 将偏置初始化为0
        nn.init.zeros_(self.reward.bias)

    def forward(self, hidden_states):
        # 给出奖励
        return self.reward(hidden_states)

class GPT2RewardHead(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_head = RewardHead(self.llm.config)

    def forward(self, input_ids, attention_mask):
        # gpt2的前向传播，但是还要输出隐藏层
        transformer_outputs = self.llm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = transformer_outputs.hidden_states[-1]
        # 给出奖励
        reward = self.reward_head(last_hidden_state).squeeze(-1)
        # sigmoid用来将奖励搞到(0,1)范围内
        return torch.sigmoid(reward)
```

初始化奖励模型

```python
model = GPT2RewardHead(model_name)
```

处理数据

```python
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
# 还是将 eos token 作为 pad token
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorWithPadding(tokenizer)
dataloader_params = {
    'batch_size': 32, # 还是使用6G显存
    'shuffle': True,
    'collate_fn': data_collator
}
train_dataloader = DataLoader(
    tokenized_dataset_train,
    **dataloader_params
)
val_dataloader = DataLoader(
    tokenized_dataset_val,
    **dataloader_params
)

batch = next(iter(train_dataloader))
print(batch.keys())

print(batch['input_ids'][1])
print(batch['attention_mask'][1])
print(batch['score'][1])
print(batch['score_index'][1])
print(tokenizer.decode(batch['input_ids'][1]))
print(batch['attention_mask'][1].nonzero()[-1])
```

模型输出

```python
outputs = model(batch['input_ids'], batch['attention_mask'])
print(outputs.shape)
```

训练相关设置

```python
device = torch.device('cuda') \
    if torch.cuda.is_available() \
    else torch.device('cpu')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# 二分类交叉熵损失
criterion = nn.BCELoss()
num_epochs = 1 # N+ Implementation Detail paper
```

评估函数

```python
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
            # 对输出进行评分
            scores = model(**model_inputs)
            # 批次中每条数据的索引
            batch_indices = torch.arange(scores.shape[0])
            # 根据索引拿出评分，也就是reward token的评分
            score = scores[batch_indices, inputs['score_index']]
            # 目标评分，0 或者 1 。
            target = inputs['score']
            # 计算误差
            loss = criterion(score, target)
        total_loss += loss.item()
    print('validation loss:', total_loss / len(val_dataloader))
```

训练循环

```python
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
        # 模型针对训练数据的打分
        scores = model(**model_inputs)
        batch_indices = torch.arange(scores.shape[0])
        # 模型对reward token的打分
        score = scores[batch_indices, inputs['score_index']]
        # 真实分数：0或者1
        target = inputs['score']
        loss = criterion(score, target)
        # 三部曲：清空梯度 ⟶ 反向传播计算梯度 ⟶ 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('损失：', loss.item())
    validate()
```

保存模型

```python
torch.save(model.state_dict(), 'reward_model.pt')
```

评估

```python
validate()
```

困惑矩阵（Confusion Matrix）

```python
from sklearn.metrics import confusion_matrix
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
```

输出结果

```
困惑矩阵:
[[364,  60],   # 第一行：真实标签为0的情况：364个预测正确，60个预测错误
 [ 31, 412]]   # 第二行：真实标签为1的情况：31个预测错误，412个预测正确
```

## 4. PPO（近端策略优化）

奖励模型的结构，和上一节的一模一样。再贴一次。

```python
import torch
from typing import Optional
from torch import nn
import numpy as np
from transformers import AutoModelForCausalLM

class RewardHead(nn.Module):
    """
    RewardHead类给GPT2实现了一个“头”，为每个输出的token返回一个标量值。
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.reward = nn.Linear(self.hidden_size, 1)
        self._post_init()

    def _post_init(self):
        nn.init.normal_(
            self.reward.weight,
            std=(1.0 / np.sqrt(self.hidden_size + 1))
        )
        nn.init.zeros_(self.reward.bias)

    def forward(self, hidden_states):
        output = hidden_states
        return self.reward(output)


class GPT2RewardModel(nn.Module):
    """
    GPT2模型加上一个“奖励头”
    """

    def __init__(self, model_name):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        # 添加奖励头
        self.reward_head = RewardHead(self.llm.config)

    def forward(
        self,
        input_ids,
        attention_mask,
    ) -> Optional[torch.FloatTensor]:
        # GPT2的输出
        transformer_outputs = self.llm.forward(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states = True,
        )

        # 获取最后一层隐藏层
        last_hidden_state = transformer_outputs.hidden_states[-1]

        # 对隐藏层给出奖励
        rewards = self.reward_head(last_hidden_state).squeeze(-1)
        # 归一化
        return torch.sigmoid(rewards)
```

加载奖励模型

```python
model_name = "./gpt2"
reward_model = GPT2RewardModel(model_name)
reward_model.load_state_dict(torch.load(
    "reward_model.pt",
    map_location='cpu'))
```

我们在使用 PPO 玩倒立摆时，分别训练了 actor 模型和 critic 模型，共两个模型。

在倒立摆游戏中，actor 模型是玩游戏的推车的策略模型。

在 PPO-RLHF 中，actor 模型是我们在 sft 过的大语言模型 gpt2-sft 。

在 PPO-RLHF 中，critic 是一个线性层（$V_{\omega}$）。用来评判 gpt2-sft 的输出的价值是多少。

```ad-danger
title: 千万别把 “奖励” 和 “价值” 搞混了！

- 奖励：环境给出的，倒立摆中不倒下给奖励 1 ，llm中要自己训练奖励模型
- 价值：训练出来的价值函数模型对llm的输出评估价值
```

而由于我们的大语言模型 gpt2-sft 也是需要更新的（更新的目的是让 llm 输出我们偏好的回答，也就是正向情感的影评）。所以我们可以把 actor 和 critic 合并到一个模型中，一起更新权重。

critic线性层在PPO-RLHF中叫做“价值头”（value head）。定义如下

```python
import torch
from typing import Optional
from torch import nn
import numpy as np
from transformers import AutoModelForCausalLM

class ValueHead(nn.Module):
    """
    ValueHead类为GPT2实现了一个“头”，会为输出的每个token返回一个标量值
    标量值就是这个token的价值，ValueHead就是评论家。
    """
    def __init__(self, config):
        super().__init__()
        # 隐藏层维度
        self.hidden_size = config.hidden_size
        # 价值函数网络的输出是标量
        self.value = nn.Linear(self.hidden_size, 1)
        self._post_init()

    def _post_init(self):
        nn.init.normal_(
            self.value.weight,
            std=(1.0 / np.sqrt(self.hidden_size + 1))
        )
        nn.init.zeros_(self.value.bias)

    def forward(self, hidden_states):
        output = hidden_states
        return self.value(output)
```

这个要更新的模型如下定义

```python
class ModelForCausalLMWithValueHead(nn.Module):
    """
    GPT2模型+一个价值头
    """
    def __init__(self, model_path):
        super().__init__()
        # 这个要初始化为我们微调出来的gpt2-sft模型
        # actor演员模型：策略模型
        self.llm = AutoModelForCausalLM.from_pretrained(model_path)
        # 添加价值头
        # critic评论家模型：价值函数模型，价值头，线性层
        self.v_head = ValueHead(self.llm.config)

    def forward(
        self,
        input_ids,
        attention_mask,
    ) -> Optional[torch.FloatTensor]:
        # gpt2-sft模型的输出
        transformer_outputs = self.llm.forward(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states = True,
        )
        # 输出的token的logits，维度为 `vocab_size`
        lm_logits = transformer_outputs.logits
        # 获取最后一层隐藏层
        last_hidden_state = transformer_outputs.hidden_states[-1]

        # 评估token的价值，评估的是最后一个隐藏层的价值
        value = self.v_head(last_hidden_state).squeeze(-1)
        # 返回输出的token的logits和token的价值
        return lm_logits, value

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)
```

初始化上面定义的模型

```python
model_path = './gpt2-sft'
model = ModelForCausalLMWithValueHead(model_path)
```

准备数据集

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

from datasets import load_dataset
dataset = load_dataset("./sst2")
print(dataset)

ds_train, ds_val = dataset['train'], dataset['validation']
```

将token数量小于 9 的文本过滤掉。

```python
print(len(ds_train))
ds_train = ds_train.filter(
    lambda x: len(x['sentence'].split(' ')) > 8)
ds_val = ds_val.filter(
    lambda x: len(x['sentence'].split(' ')) > 8)

print(len(ds_train))
print(len(ds_val))
```

控制输入的token数量

```python
import random
input_min_token_length = 2
input_max_token_length = 8
input_token_length_range = list(range(
    input_min_token_length,
    input_max_token_length))
print(input_token_length_range)
print(random.choice(input_token_length_range))
```

对数据集分词

```python
def tokenize(sample):
    # 提示词token的数量随机选择一个
    input_size = random.choice(input_token_length_range)
    # 如果input_size=3，截取sentence字段文本的前3个token出来
    sample['input_ids'] = tokenizer.encode(sample['sentence'])[:input_size]
    # 前3个token掩码为1
    sample['attention_mask'] = [1] * len(sample['input_ids'])
    # 前3个token对应的文本
    sample['query'] = tokenizer.decode(sample['input_ids'])
    return sample

map_kwargs = {
    "batched": False,
    "remove_columns": ['idx', 'sentence', 'label']
}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)
tokenized_dataset_val = ds_val.map(tokenize, **map_kwargs)
```

设置 PyTorch 数据格式

```python
tokenized_dataset_train.set_format(type='torch')
tokenized_dataset_val.set_format(type='torch')

print(tokenized_dataset_train[6])
```

对 `REWARD_TOKEN_ID` 进行奖励。

```python
REWARD_TOKEN_ID = tokenizer.eos_token_id
```

构建数据加载器dataloader

```python
from torch.utils.data import DataLoader

batch_size = 32

def collator(batch):
    return dict((key, [d[key] for d in batch]) for key in batch[0])

train_dataloader = DataLoader(
    tokenized_dataset_train,
    batch_size=batch_size,
    collate_fn=collator,
    shuffle=True
)
val_dataloader = DataLoader(
    tokenized_dataset_val,
    batch_size=batch_size,
    collate_fn=collator,
    shuffle=True
)

batch = next(iter(train_dataloader))
print(batch)
```

控制gpt2-sft输出的token数量

```python
output_min_length = 5
output_max_length = 16

# https://huggingface.co/docs/trl/how_to_train#how-to-generate-text-for-training
# gpt2-sft输出的配置
# - 模型会从整个词汇表中按照原始概率分布进行采样
# - 每个词被选中的概率完全由模型的原始输出决定
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0, # 所有词汇表中的词都可能被选中
    "top_p": 1.0, # 包含整个概率分布
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id
}
```

采集样本

```python
# 随机去一个生成长度
new_tokens = random.choice(list(range(
    output_min_length,
    output_max_length)))
# 设置生成长度这个参数
generation_kwargs["max_new_tokens"] = new_tokens
sample = tokenizer('Hi, this')
print(sample)
```

看一下 `"Hi, this"` 输入到大模型后的输出是什么？

```python
query_response = model.generate(
    input_ids=torch.tensor(sample['input_ids']).unsqueeze(0),
    attention_mask=torch.tensor(sample['attention_mask']).unsqueeze(0),
    **generation_kwargs
).squeeze(0)
print(query_response)
```

![[squeeze和unsqueeze用法示意图.png]]

解码一下输出的token

```python
print(tokenizer.decode(query_response))
```

看一下输出的token的奖励能拿多少？

```python
with torch.no_grad():
    # 首先添加reward token
    query_response_score = torch.cat([
        query_response,
        torch.tensor([REWARD_TOKEN_ID])])
    attention_mask = torch.ones_like(
        query_response_score,
        dtype=torch.long)
    score = reward_model(
        query_response_score.unsqueeze(0),
        attention_mask.unsqueeze(0)
    ).squeeze(0)[-1]
print(score)
```

生成一批数据看一下

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
reward_model = reward_model.to(device)

query_tensors = batch['input_ids'] # 提示词的张量
query_attention_masks = batch['attention_mask'] # 提示词的掩码张量

response_tensors = [] # 补全的张量
query_response_tensors = [] # [提示词+补全] 张量
score_tensors = [] # 奖励模型给的分数的张量

for i, query in enumerate(query_tensors):
    query = query.to(device)
    query_attention_mask = query_attention_masks[i].to(device)
    new_tokens = random.choice(list(range(
        output_min_length,
        output_max_length)))
    generation_kwargs["max_new_tokens"] = new_tokens
    query_response = model.generate(
        input_ids=query.unsqueeze(0),
        attention_mask=query_attention_mask.unsqueeze(0),
        **generation_kwargs
    ).squeeze(0)
    # 补全的长度 = （提示词+补全的长度） - 提示词的长度
    response_len = len(query_response) - len(query)
    # 截取出补全的张量
    response_tensors.append(query_response[-response_len:])
    query_response_tensors.append(query_response)

    with torch.no_grad():
        query_response_score = torch.cat([
            query_response, # 完整的提示词+补全张量
            torch.tensor([REWARD_TOKEN_ID]).to(device)])
        attention_mask = torch.ones_like(
            query_response_score, dtype=torch.long)
        score = reward_model(
            query_response_score.unsqueeze(0),
            attention_mask.unsqueeze(0)
        ).squeeze(0)[-1]
        # 将奖励模型给的分数处理一下
        score = 2 * (score - 0.5)
    score_tensors.append(score)

batch["response"] = [
    tokenizer.decode(response) for response in response_tensors
]
from pprint import pprint
pprint(batch['response'])
```

计算奖励

$$
\begin{aligned}
\text{reward} &= \text{score} - \beta\log\left(\frac{\pi_\theta^{\text{RL}}}{\pi^{\text{SFT}}}\right) \\ &= \text{score} - \beta(\log\pi_\theta^{\text{RL}}-\log\pi^{\text{SFT}})
\end{aligned}
$$

![[ppo-reward-calculate.excalidraw|1000]]

冻结一个我们sft后的gpt2，也就是gpt2-sft，当然也包括ValueHead。

```python
from copy import deepcopy
sft_model = deepcopy(model)
```

处理数据，添加 padding

```python
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

input_data = data_collator([
    {'input_ids': ids,
     'attention_mask': torch.ones_like(ids)} \
         for ids in query_response_tensors
]).to(device)
print(input_data)
```

实现计算奖励的函数

```python
def compute_rewards(
    input_data, # 输入数据
    query_tensors, # 提示词张量
    response_tensors, # 补全的张量
    score_tensors # 奖励模型给出的分数的张量
):
    with torch.no_grad():
        # 正在微调的模型所输出的token的logits和token的价值
        logits, values = model(**input_data) # b, seq, vocab_size
        # 冻结的模型的输出
        ref_logits, _ = sft_model(**input_data)
        # 正在微调的模型的输出的对数概率 `log_softmax`
        # 去掉最后一个token，因为是预测下一个token的任务
        # input_data如果是："abcde"，那么建立的数据对为：
        # abcd --> bcde
        logp = torch.nn.functional.log_softmax(
            logits[:, :-1, :],
            dim=-1
        )
        # 冻结的模型的输出的对数概率
        ref_logp = torch.nn.functional.log_softmax(
            ref_logits[:, :-1, :],
            dim=-1
        )
        # 实际生成的token序列
        # 自回归模型是预测下一个token，所以去掉第一个token
        # 真实标签为：bcde，需要去掉a
        labels = input_data['input_ids'][:, 1:] # b, seq
        # 使用gather提取实际token的概率
        # logp 是 vocab_size 大小的张量
        # 假设真实的label是 `hello`
        # 那么要取出 `hello` 在 logp 张量中的概率
        logp = torch.gather(
            logp,
            2,
            labels.unsqueeze(-1)
        ).squeeze(-1) # batch, seq
        ref_logp = torch.gather(
            ref_logp,
            2,
            labels.unsqueeze(-1)
        ).squeeze(-1) # batch, seq
        # kl散度
        kl = logp - ref_logp
        # kl散度的权重
        beta = 0.2
        # 最终奖励的计算
        rewards = - beta * kl
        attention_mask = input_data['attention_mask']
        # 预测下一个token，所以去掉第一个mask
        masks = torch.zeros_like(attention_mask[:, 1:])
        masks[:,:] = attention_mask[:, 1:]
        # 遍历批次中的每一个提示词张量
        for j in range(len(query_tensors)):
            # 补全开始的索引
            start = len(query_tensors[j]) - 1
            # 补全结束的索引
            end = start + len(response_tensors[j])
            # 提示词部分掩码为0
            masks[j, :start] = 0
            # 补全后面的填充token掩码为0
            masks[j, end:] = 0
            # 将奖励模型给出的分数加到补全的最后一个token的奖励上面
            rewards[j, end - 1] += score_tensors[j]
            # 只留下掩码为1的部分的奖励
            rewards[j, :] *= masks[j, :]
            # 只留下掩码为1的部分的价值
            values[j, :-1] *= masks[j, :]

    return logp, rewards, values[:, :-1], masks
```

![[ppo-reward.excalidraw|1000]]


测试一下计算奖励的函数

```python
logprobs, rewards, values, masks = compute_rewards(
    input_data,
    query_tensors,
    response_tensors,
    score_tensors
)
print(rewards[0])
print(input_data['input_ids'][0])
print(input_data['attention_mask'][0])
print(masks[0])
print(values[0])
```

计算优势（广义优势估计）

![[gae.excalidraw|1000]]


```python
def masked_mean(values, mask):
    # 计算带掩码的平均值
    return (values * mask).sum() / mask.sum()

def masked_var(values, mask):
    # 计算带掩码的方差
    mean = masked_mean(values, mask)
    centred_values = values - mean
    return masked_mean(centred_values ** 2, mask)

def masked_whiten(values, mask):
    '''
    对数据进行带掩码的白化处理，
    让有效数据的方差变为1，但均值保持不变
    '''
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    whitened += mean
    return whitened

def compute_advantage(rewards, values, masks):
    '''
    广义优势估计（GAE）
    '''
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
    # 对广义优势估计进行了白化处理
    advantages = masked_whiten(advantages, masks)

    returns = advantages + values
    return advantages, returns
```

测试一下计算优势的函数

```python
advantages, returns = compute_advantage(rewards, values, masks)
print(advantages[0])
print(returns[0])
```

小批次PPO训练

训练配置

```python
learning_rate = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# 随机排列一下各个批次大小
np.random.permutation(batch_size)
```

开始训练！

```python
# 最小的批次大小
mini_batch_size = 4
# 训练 4 个 epoch
ppo_epochs = 4
# ε = 0.2
cliprange_ratio = 0.2

v_loss_coeff = 0.1
# 比例的阈值
ratio_threshold = 10

def compute_loss(
    old_logprobs, # 冻结的一份概率
    values, # 价值
    logprobs, # 正在微调的模型输出的对数概率
    vpreds, # 预测的价值
    masks, # 掩码
    advantages, # 广义优势估计
    returns # 回报
):
    # 比率
    ratio = torch.exp(logprobs - old_logprobs)
    # 比率 * 广义优势估计
    pg_loss1 = - ratio * advantages
    # clip(比率，1-ϵ,1+ϵ) * 广义优势估计
    pg_loss2 = - torch.clamp(
        ratio,
        1 - cliprange_ratio,
        1 + cliprange_ratio
    ) * advantages
    # 策略（gpt2-sft）的损失
    pg_loss = masked_mean(torch.max(pg_loss1, pg_loss2), masks)
    # 价值网络（价值头）的损失，mse
    v_loss = masked_mean((vpreds - returns) ** 2, masks)
    # 由于 正在微调的模型 = gpt2-sft + value_head
    # 总的损失 = 策略网络的损失 + 0.1 * 价值网络的损失
    loss = pg_loss + v_loss_coeff * v_loss
    # 计算平均比率
    avg_ratio = masked_mean(ratio, masks)
    # 这一步不在ppo公式中
    # 如果平均比率 > 10
    if avg_ratio > ratio_threshold:
        pg_loss = pg_loss * 0.0
        v_loss = v_loss * 0.0
        loss = loss * 0.0

    return loss, v_loss

def mini_batch_train():
    for ep in range(ppo_epochs):
        batch_inds = np.random.permutation(batch_size)
        # range(0, 32, 4)
        for start in range(0, batch_size, mini_batch_size):
            # start = 0; end = 4
            end = start + mini_batch_size
            mini_batch_inds = batch_inds[start:end]

            mb_model_inputs = {
                'input_ids': input_data \
                    ['input_ids'] \
                    [mini_batch_inds],
                'attention_mask': input_data \
                    ['attention_mask'] \
                    [mini_batch_inds]
            }
            # 模型的输出是token的logits和value
            mb_logits, mb_vpreds = model(**mb_model_inputs)
            # 去掉最后一个token
            mb_logits = torch.nn.functional.log_softmax(
                mb_logits[:, :-1, :],
                dim=-1
            )
            # 取出真实标签对应的概率
            mb_logprobs = torch.gather(
                mb_logits,
                2,
                mb_model_inputs['input_ids'][:, 1:].unsqueeze(-1)
            ).squeeze(-1)

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
```

![[ppo-loss.excalidraw|1000]]


测试一下这个训练函数

```python
mini_batch_train()
```

正式开始训练

```python
num_epochs = 1

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 生成补全内容（回复）
        query_tensors = batch['input_ids'] # 提示词的张量
        query_attention_masks = batch['attention_mask']

        response_tensors = [] # 补全的张量
        query_response_tensors = [] # 提示词+补全的张量
        score_tensors = [] # 分数的张量

        for i, query in enumerate(query_tensors):
            query = query.to(device)
            query_attention_mask = query_attention_masks[i].to(device)
            # 随机挑一个补全的长度
            new_tokens = random.choice(list(range(
                output_min_length,
                output_max_length)))
            # 设置补全长度属性
            generation_kwargs["max_new_tokens"] = new_tokens
            # 提示词 + 补全
            query_response = model.generate(
                input_ids=query.unsqueeze(0),
                attention_mask=query_attention_mask.unsqueeze(0),
                **generation_kwargs
            ).squeeze(0)
            # 补全的长度
            response_len = len(query_response) - len(query)
            # 补全的张量
            response_tensors.append(query_response[-response_len:])
            query_response_tensors.append(query_response)

            with torch.no_grad():
                # 提示词 + 补全 + reward_token
                query_response_score = torch.cat([
                    query_response,
                    torch.tensor([REWARD_TOKEN_ID]).to(device)])
                attention_mask = torch.ones_like(
                    query_response_score,
                    dtype=torch.long)
                # 奖励模型的评分
                score = reward_model(
                    query_response_score.unsqueeze(0),
                    attention_mask.unsqueeze(0)
                ).squeeze(0)[-1]
                # 将奖励模型的评分从(0,1)缩放到(-1,1)
                score = 2 * (score - 0.5)
            score_tensors.append(score)

        input_data = data_collator([
            {
                'input_ids': ids,
                'attention_mask': torch.ones_like(ids)
            }
            for ids in query_response_tensors
        ]).to(device)

        # 奖励和优势
        logprobs, rewards, values, masks = compute_rewards(
            input_data,
            query_tensors,
            response_tensors,
            score_tensors
        )
        advantages, returns = compute_advantage(rewards, values, masks)

        # 小批次训练
        mini_batch_train()
    print(f'epoch {epoch + 1} finished')
```

验证

```python
print(len(tokenized_dataset_val))
val_gen_lengths = [0] * len(tokenized_dataset_val)
for i in range(len(tokenized_dataset_val)):
    val_gen_lengths[i] = random.choice(list(range(
        output_min_length,
        output_max_length)))
val_gen_lengths[:10]
```

验证函数的编写

```python
def validate():
    scores = []
    for b, batch in enumerate(val_dataloader):
        # 生成补全内容
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
            query_response_score = torch.cat([
                query_response,
                torch.tensor([REWARD_TOKEN_ID]).to(device)])
            attention_mask = torch.ones_like(
                query_response_score, dtype=torch.long)
            score = reward_model(
                query_response_score.unsqueeze(0),
                attention_mask.unsqueeze(0)
            ).squeeze(0)[-1]
            score = 2 * (score - 0.5)
            scores.append(score.item())
    print('平均分数:', sum(scores) / len(scores))
```

验证一下PPO微调后的效果

```python
validate()
```

保存模型

```python
torch.save(model.state_dict(), 'gpt2-ppo.pt')
```

测试一下gpt2-sft模型的输出

```python
model_path = './gpt2-sft'
model = ModelForCausalLMWithValueHead(model_path).to(device)
validate()
```

```ad-danger
title: PPO的缺陷

就 PPO 算法而言，可以说是完美！主要问题在于很消耗 GPU 。同时需要运行 4 个模型。

- actor: gpt2-sft
- critic: value head
- reward model: 冻结的奖励模型（这个还需要单独训练，很难训练！）
- 冻结的参考模型gpt2-sft
```

