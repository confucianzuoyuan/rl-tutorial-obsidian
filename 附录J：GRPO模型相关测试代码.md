
```python
from qwen2_model import Transformer
import torch
from pathlib import Path
from tokenizer import Tokenizer
from countdown_task import CountdownTasksDataset, reward_function
from grpo import rollout
from torch.utils.data import DataLoader
from pprint import pprint
t = Tokenizer("./Qwen2.5-0.5B-Instruct/tokenizer.json")
c = CountdownTasksDataset(tokenizer=t, data_path="Countdown-Tasks-3to4")
pprint(c.encode_prefix(numbers=[1, 2, 3], target=6)["prefix"])
print(c.encode_prefix(numbers=[1, 2, 3], target=6)["prefix_tokens"])
print(c.encode_prefix(numbers=[1, 2, 3], target=6)["prefix_token_ids"])

response = """
<think>我认为应该。。。</think>
<answer>1+2+3</answer>
"""

pprint(reward_function(response=response, numbers=[1, 2, 3], target=6))


pretrained_model_path = Path("./Qwen2.5-0.5B-Instruct")
device = torch.device("cuda")
dtype = torch.bfloat16

torch.set_default_device(device)
torch.random.manual_seed(1337)
BATCH_SIZE = 4
NUM_QUESTIONS_PER_BATCH = 2
NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH


train_dataset = CountdownTasksDataset(
    data_path="Countdown-Tasks-3to4",
    tokenizer=t,
    split="train",
    test_size=128,
)
generator = torch.Generator(device=device)
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=CountdownTasksDataset.collate_fn,
    generator=generator,
    batch_size=NUM_QUESTIONS_PER_BATCH,
)

print(train_dataloader)
b = next(iter(train_dataloader))
print(b.prefix[0])

model = Transformer.from_pretrained(
    "Qwen2.5-0.5B-Instruct", device=device).train()

episodes = rollout(
    model=model,
    tokenizer=t,
    batch=b,
    max_gen_len=1024,
    num_answer_per_question=2,
    reward_function=reward_function,
    device=device,
    dtype=dtype,
)
print("\n=========采集轨迹===========")
for episode in episodes:
    print("=============prefix start=============")
    pprint(episode.prefix)
    print("=============prefix end===============")
    print("===============text=======================")
    pprint(episode.text)
    print("===============text=======================")
    print("===============text=======================")
    print("===============text=======================")
    print("===============text=======================")
    print("===============text=======================")
```

```
('<|im_start|>system\n'
 '你是一个有用的助手。你首先在脑海中思考推理过程，然后为用户提供答案。<|im_end|>\n'
 '<|im_start|>user\n'
 '使用这些数字 [1, 2, 3]，创建一个等于 6 的等式。你可以使用基本算术运算（+、-、*、/），每个数字只能使用一次。在 <think> '
 '</think> 标签中展示你的解题过程。并在 <answer> </answer> 标签中返回最终答案，例如 <answer> (1 + 2) / 3 '
 '</answer>。<|im_end|>\n'
 '<|im_start|>assistant\n'
 '让我一步步来解决这个问题。\n'
 '<think>')
```

