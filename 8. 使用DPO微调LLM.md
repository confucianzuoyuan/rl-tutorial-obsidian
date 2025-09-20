
## 1. 预训练模型

我们使用 `Qwen2.5-0.5B` 作为底座模型。

```ad-danger
title: 注意不是 `Qwen2.5-0.5B-Instruct` 模型

这个是经过指令微调的
```

推理脚本 `infer.py`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen2.5-0.5B",
    torch_dtype="auto",
    device_map="auto",
)

model.generation_config.do_sample = True
model.generation_config.eos_token_id = [151645, 151643]
model.generation_config.pad_token_id = 151643
model.generation_config.temperature = 0.7
model.generation_config.top_p = 0.8
model.generation_config.top_k = 20
model.generation_config.repetition_penalty = 1.05

tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B")

history = []
# history.append({"role": "system", "content": "You are a helpful assistant"})
while True:
    question = input('User：' + '\n')
    print('\n')
    history.append({"role": "user", "content": question})

    input_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )
    model_inputs = tokenizer([input_text], return_tensors="pt").to(device)

    if model_inputs.input_ids.size()[1] > 32000:
        break

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1000
    )

    if len(generated_ids) > 32000:
        break

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('Assistant:\n')
    print(response)
    print("--------------------")
    print('\n')
    history.append({"role": "assistant", "content": response})

print("超过模型字数上线，已退出")
```

## 2. 基于指令的SFT

**Instruct-SFT优化目标**

![[sft-loss.excalidraw|1000]]

**损失函数的实现**

![[ppo-loss-impl.excalidraw|1000]]

**推广到批次**

![[ppo-batch-loss.excalidraw|1000]]

### 代码实现

加载qwen2.5-0.5B基座模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import os

device = "cuda"
model_path = "./Qwen2.5-0.5B"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

重设模型的 `generation_config` 文件，以便最后对比训练前后生成式问答的效果

```python
print(model.generation_config)

model.generation_config.do_sample = True
model.generation_config.eos_token_id = [151645, 151643]
model.generation_config.pad_token_id = 151643
model.generation_config.temperature = 0.7
model.generation_config.top_p = 0.8
model.generation_config.top_k = 20
model.generation_config.repetition_penalty = 1.05

print(model.generation_config)
```

定义SFT阶段模型训练超参

```python
from dataclasses import dataclass
@dataclass
class SFTConfig:
    max_length:int = 2500
    batch_size:int = 2
    gradient_accumulation_steps:int = 8
    log_iter:int = 400
    max_lr:float = 2e-5
    min_lr:float = 2e-6
    warmup_steps:int = 1000
```

导入训练数据

```python
import datasets
ultrachat_200k_data = datasets.load_dataset('./ultrachat_200k')
```

这是一套比较好的 **英文** 多轮对话数据。

训练数据转化成 `tokenid`：`str -> tokenid`

```python
def tokenize_and_format(data):
    input_ids = tokenizer.apply_chat_template(
        data,
        tokenize = True,
        add_generation_prompt = False,
        truncation = True,
        max_length = 2500,
    )
    
    return input_ids

## 生成训练数据的tokenid
chosen_input_ids_list = []
i = 0
while True:
    data = ultrachat_200k_data['train_sft'][i]['messages']
    # 添加 **系统提示词**
    data.insert(
        0,
        {"content": "You are a helpful assistant", "role": "system"}
    )
    input_ids = tokenize_and_format(data)
    chosen_input_ids_list.append(input_ids)
    i += 1
    if i % 1000 == 0:
        print(f"已处理{i}条数据")
    if i == 50000:#len(ultrachat_200k_data['train_sft']):
        break
print('-' * 70)
```

使用设置的训练超参数

```python
batch_size = SFTConfig.batch_size
gradient_accumulation_steps = SFTConfig.gradient_accumulation_steps
log_iter = SFTConfig.log_iter
max_lr = SFTConfig.max_lr
min_lr = SFTConfig.min_lr
warmup_steps = SFTConfig.warmup_steps
total_steps = len(chosen_input_ids_list) // batch_size
optimizer = torch.optim.AdamW(filter(
    lambda p: p.requires_grad,
    model.parameters()
), lr=max_lr)
trainable_parameters_num = sum(p.numel() for p in filter(
    lambda p: p.requires_grad,
    model.parameters())
)  ##全参微调
```

配置logging日志记录模型训练过程

```python
##配置logging
import time

with open(f"./Qwen2.5-0.5B-SFT_log.txt", "a") as my_file:
    my_file.write(f' \
        time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, \
        batch_size:{batch_size}, \
        trainable_parameters_num:{trainable_parameters_num}, \
        warmup_steps:{warmup_steps}, \
        max_lr:{max_lr}, \
        min_lr:{min_lr}\n')

#定义一个日志记录函数
def log_call(iters, iters_average_loss):
    with open(f"./Qwen2.5-0.5B-SFT_log.txt", "a") as my_file:
        my_file.write(f' \
            time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, \
            iters:{iters+1}, \
            iters_average_Loss:{iters_average_loss:.4f}\n')
```

学习率设置：余弦衰减学习率

![[lr-decay.excalidraw|1000]]
```python
def linear_warmup(current_step, warmup_steps, max_lr):
    if current_step < warmup_steps:
        return max_lr * current_step / warmup_steps
    else:
        return max_lr

def cosine_decay(current_step, warmup_steps, total_steps, max_lr, min_lr):
    if current_step < warmup_steps:
        return linear_warmup(current_step, warmup_steps, max_lr)
    else:
        progress = (current_step - warmup_steps)          \
                   /                                      \
                   (total_steps - warmup_steps)
        decay = 0.5 * (1 + np.cos(np.pi * progress))
        return (max_lr - min_lr) * decay + min_lr
```

掩码设置

- SFT和预训练的区别核心就是掩码掉“问题”部分的损失，而 ==只看“回答”部分的损失==，并仅基于回答部分的损失进行优化
- 实现方式：==构造损失掩码==，仅针对每轮对话（含多轮）的模型“输出”部分（也就是回答部分）进行损失计算

![[sft-dialogue.excalidraw|1000]]


```python
def create_answer_mask(input_ids, tokenizer):
    """
    创建仅对助手回答部分计算损失的掩码
    
    Args:
        input_ids: 输入token序列 [batch_size, seq_len]
        tokenizer: 分词器
    
    Returns:
        answer_mask: 助手回答部分为1，其他部分为0的掩码
    """
    batch_size, seq_len = input_ids.shape
    answer_mask = torch.zeros_like(input_ids)
    
    # 获取结束标记的token id
    eos_token_id = tokenizer.encode('<|im_end|>')[0]
    
    for batch_idx in range(batch_size):
        # 找到所有 <|im_end|> 的位置
        eos_positions = torch.where(
            input_ids[batch_idx] == eos_token_id
        )[0].tolist()
        
        if len(eos_positions) < 2:  # 至少需要user和assistant各一个结束标记
            continue
            
        # 解析对话轮次
        user_ends, assistant_ends = \
            _parse_conversation_turns(eos_positions)
        
        # 为每个助手回答设置掩码
        _set_answer_masks(
            answer_mask[batch_idx],
            user_ends,
            assistant_ends,
            seq_len
        )
    
    return answer_mask


def _parse_conversation_turns(eos_positions):
    """
    解析对话轮次，分离用户和助手的结束位置
    
    对话格式：
    <|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>\n
    
    eos_positions[0]: system结束 (如果有)
    eos_positions[1]: 第1轮user结束
    eos_positions[2]: 第1轮assistant结束
    eos_positions[3]: 第2轮user结束
    eos_positions[4]: 第2轮assistant结束
    ...
    """
    # 跳过system系统提示词部分，从第一个user开始
    conversation_eos = eos_positions[1:]  # 去掉system的<im_end>
    
    # 偶数索引：user结束位置，奇数索引：assistant结束位置
    user_ends = [pos + 1 for pos in conversation_eos[::2]] # 每隔2个取一个，从0开始
    assistant_ends = [pos + 1 for pos in conversation_eos[1::2]] # 每隔2个取一个，从1开始
    
    return user_ends, assistant_ends


def _set_answer_masks(mask, user_ends, assistant_ends, seq_len):
    """
    为助手回答部分设置掩码
    
    Args:
        mask: 当前样本的掩码 [seq_len]
        user_ends: 用户消息结束位置列表
        assistant_ends: 助手消息结束位置列表
        seq_len: 序列长度
    """
    num_user_turns = len(user_ends)
    num_assistant_turns = len(assistant_ends)
    
    if num_user_turns == num_assistant_turns:
        # 完整对话：每轮都有用户问题和助手回答
        for user_end, assistant_end in zip(user_ends, assistant_ends):
            answer_start = user_end + 3  # 跳过 <|im_start|>assistant\n
            answer_end = assistant_end - 1  # 不包含 <|im_end|>
            mask[answer_start:answer_end] = 1
            
    elif num_user_turns == num_assistant_turns + 1:
        # 未完成对话：最后一轮助手回答被截断
        
        # 处理完整的对话轮次
        for user_end, assistant_end in zip(user_ends[:-1], assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1
        
        # 处理最后一轮被截断的助手回答
        last_user_end = user_ends[-1]
        last_answer_start = last_user_end + 3
        mask[last_answer_start:] = 1  # 到序列结尾
```

开启SFT微调训练

```python
model.train()
training_losses = []
model.zero_grad()  # 训练开始时清空梯度
skipped_batches_count = 0

total_batches = len(chosen_input_ids_list) // batch_size

for batch_idx in range(total_batches):
    ## ==================== 数据准备阶段 ====================
    
    # 获取当前批次的原始数据
    current_batch_sequences = chosen_input_ids_list[
        batch_idx * batch_size : (batch_idx + 1) * batch_size
    ]
    
    # 计算当前批次的最大序列长度，用于padding对齐
    max_sequence_length = max([len(sequence) for sequence in current_batch_sequences])
    
    ### 对批次数据进行右填充，使所有序列长度一致以便并行计算
    padded_sequences_list = []
    pad_token_id = model.generation_config.eos_token_id[-1]
    
    for seq_idx in range(batch_size):
        # 原始的一条训练数据
        original_sequence = current_batch_sequences[seq_idx]
        # 要填充的长度
        padding_length = max_sequence_length - len(original_sequence)
        
        # 使用EOS token进行右填充
        padded_sequence = torch.nn.functional.pad(
            torch.tensor(original_sequence),
            (0, padding_length),
            mode='constant',
            value=pad_token_id
        ).tolist()
        
        padded_sequences_list.append(padded_sequence)
    
    # 转换为张量
    batch_input_tensor = torch.tensor(padded_sequences_list)

    ## ==================== 构建输入输出对 ====================
    
    # 构建因果语言模型的输入输出对：x->y（下一个词预测）
    model_inputs = batch_input_tensor[:, :-1].to(device)    # 输入：前n-1个token
    target_labels = batch_input_tensor[:, 1:].to(device)    # 标签：后n-1个token

    ## ==================== 构建训练掩码 ====================
    
    # 构建掩码矩阵来控制损失计算范围
    # 1. padding_mask：标识哪些位置是填充token（不计算损失）
    # 2. answer_mask：标识哪些位置是助手回答部分（只对回答计算损失）
    
    ### 【填充掩码】：非填充token为1，填充token为0
    ### padding_mask中的问题部分的掩码也是1
    padding_mask = torch.where(target_labels == pad_token_id, 0, 1)
    
    ### 【回答掩码】：只有助手回答部分为1，其他部分为0
    assistant_answer_mask = create_answer_mask(model_inputs, tokenizer)
    
    ### 【组合掩码】：同时满足"非填充"且"是回答部分"的token才计算损失
    ### 取出交集，就是真正要计算的回答部分
    final_loss_mask = (assistant_answer_mask & padding_mask)

    ## ==================== 批次有效性检查 ====================
    
    # 检查当前批次是否有效：如果某个样本的回答部分完全为空，则跳过该批次
    # 这种情况通常发生在问题过长导致回答部分被截断时
    tokens_per_sample = final_loss_mask.sum(dim=-1)  # 每个样本的有效回答token数
    min_answer_tokens = tokens_per_sample.min().item()  # 最少的有效token数
    
    if min_answer_tokens == 0:
        print(f'⚠️ 跳过第{batch_idx + 1}批次：回答部分数据不足')
        skipped_batches_count += 1
        continue  # 跳过当前批次

    ## ==================== 模型前向传播 ====================
    
    # 执行前向传播，获取模型预测的logits
    # [batch_size, seq_length, vocab_size]
    model_logits = model(model_inputs).logits
    
    ## ==================== 损失计算 ====================
    
    # 计算带掩码的交叉熵损失
    # 步骤：logits -> softmax -> log -> gather -> 负对数似然 -> 掩码过滤 -> 平均
    
    # 1. 计算每个token的负对数似然损失，
    # 形状：[batch_size, seq_len, vocab_size]
    log_probabilities = torch.log(torch.softmax(model_logits, dim=-1))
    # 使用真正的目标token取出vocab_size长度的数组中token对应的对数概率
    # 形状：[batch_size, seq_len]
    gathered_log_probs = torch.gather(
        log_probabilities,
        dim=-1,
        index=target_labels.unsqueeze(2)
    )
    negative_log_likelihood = gathered_log_probs * (-1)  # 负对数似然
    token_losses = negative_log_likelihood.squeeze(2)
    
    # 2. 应用掩码并计算每个样本的平均损失
    masked_token_losses = torch.mul(token_losses, final_loss_mask)
    sample_losses = masked_token_losses.sum(dim=-1) \
                  / final_loss_mask.sum(dim=-1)
    
    # 3. 计算批次平均损失并应用梯度累积
    batch_average_loss = torch.nanmean(sample_losses) \
                       / gradient_accumulation_steps

    ## ==================== 反向传播和优化 ====================
    
    # 反向传播计算梯度
    batch_average_loss.backward()

    # 动态调整学习率（余弦衰减 + 预热）
    current_learning_rate = cosine_decay(
        batch_idx,
        warmup_steps,
        total_steps,
        max_lr,
        min_lr
    )
    
    # 更新优化器的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_learning_rate

    # 梯度累积：只在累积步数达到或最后一个批次时更新权重
    is_accumulation_step = (batch_idx + 1) \
                         % gradient_accumulation_steps == 0
    is_final_batch = (batch_idx + 1) == total_batches
    
    if is_accumulation_step or is_final_batch:
        optimizer.step()        # 更新模型权重
        optimizer.zero_grad()   # 清空梯度缓存

    ## ==================== 训练日志记录 ====================
    
    # 记录当前批次的损失（还原梯度累积的缩放）
    actual_batch_loss =                   \
        batch_average_loss.item()         \
        *                                 \
        gradient_accumulation_steps
    training_losses.append(actual_batch_loss)

    # 定期输出训练进度
    should_log = (batch_idx + 1) % log_iter == 0 or is_final_batch
    
    if should_log:
        # 计算最近几个批次的平均损失
        recent_losses = training_losses[-log_iter:]
        recent_average_loss = np.nanmean(recent_losses)
        
        # 输出训练状态
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f'⏰ 时间: {current_time} | '
              f'📊 批次: {batch_idx + 1}/{total_batches} | '
              f'📈 最近{len(recent_losses)}批次平均损失: {recent_average_loss:.4f} | '
              f'🎯 学习率: {current_learning_rate:.2e}')
        
        # 调用外部日志记录函数
        log_call(batch_idx, recent_average_loss)

## ==================== 训练完成总结 ====================

print("🎉 训练完成!")
print(f'📊 训练统计:')
print(f'   - 总批次数: {total_batches}')
print(f'   - 跳过批次数: {skipped_batches_count}')
print(f'   - 有效批次数: {total_batches - skipped_batches_count}')
print(f'   - 最终平均损失: {np.nanmean(training_losses[-100:]):.4f}')

if skipped_batches_count > 0:
    skip_ratio = skipped_batches_count / total_batches * 100
    print(f'⚠️ 跳过批次占比: {skip_ratio:.2f}%')
    if skip_ratio > 10:
        print('💡 建议: 跳过批次过多，考虑增加最大序列长度或优化数据预处理')
```

保存模型

```python
model.save_pretrained('./Qwen2.5-0.5B-SFT/')
tokenizer.save_pretrained('./Qwen2.5-0.5B-SFT/')
```

## 3. 使用DPO微调SFT后的大模型

加载SFT后的模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import os

device = "cuda"
model_path = "./Qwen2.5-0.5B-SFT"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
# 冻结的参考模型
ref_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

生成回答使用同样的配置

```python
print(model.generation_config)

model.generation_config.do_sample = True
model.generation_config.eos_token_id = [151645, 151643]
model.generation_config.pad_token_id = 151643
model.generation_config.temperature = 0.7
model.generation_config.top_p = 0.8
model.generation_config.top_k = 20
model.generation_config.repetition_penalty = 1.05

print(model.generation_config)
```

定义模型训练超参数

```python
from dataclasses import dataclass
@dataclass
class DPOConfig:
    max_length:int = 1700 #根据自身具备的算力条件进行自适应更改
    batch_size:int = 2
    gradient_accumulation_steps:int = 8
    beta:float = 0.5 # β是dpo公式中的超参数
    log_iter:int = 200
    max_lr:float = 1e-6
    min_lr:float = 1e-7
    warmup_steps:int = 300
```

导入DPO偏好优化所需的训练数据

```python
import datasets
binarized_data = datasets.load_dataset('./ultrafeedback_binarized')
```

对偏好数据转化成模型接受的格式：`str -> input_ids`

```python
def tokenize_and_format(data):
    input_ids = tokenizer.apply_chat_template(
        data,
        tokenize = True,
        add_generation_prompt = False,
        truncation = True,
        max_length = DPOConfig.max_length,
    )

    return input_ids

## 生成偏好数据的input_ids
chosen_input_ids_list = []
i = 0
while True:
    data = binarized_data['train_sft'][i]['chosen']
    data.insert(
        0,
        {"content": "You are a helpful assistant", "role": "system"}
    )
    input_ids = tokenize_and_format(data)
    chosen_input_ids_list.append(input_ids)
    i += 1
    if i % 10000 == 0 or i == len(binarized_data['train_sft']):
        print(f"偏好数据已处理{i}条数据")
    if i == 30000:
        break
print('-' * 70)

#############################################################################
## 生成不偏好数据的input_ids
rejected_input_ids_list = []
i = 0
while True:
    data = binarized_data['train_sft'][i]['rejected']
    data.insert(
        0,
        {"content": "You are a helpful assistant", "role": "system"}
    )
    input_ids = tokenize_and_format(data)
    rejected_input_ids_list.append(input_ids)
    i += 1
    if i % 10000 == 0 or i == len(binarized_data['train_sft']):
        print(f"非偏好数据已处理{i}条数据")
    if i == 30000:
        break

## 确保数据条数一致
assert len(chosen_input_ids_list) == len(rejected_input_ids_list)
```

使用设置的训练超参数

```python
beta = DPOConfig.beta # β超参数
batch_size = DPOConfig.batch_size
gradient_accumulation_steps = DPOConfig.gradient_accumulation_steps
log_iter = DPOConfig.log_iter
max_lr = DPOConfig.max_lr
min_lr = DPOConfig.min_lr
warmup_steps = DPOConfig.warmup_steps
total_steps = len(chosen_input_ids_list) // batch_size
optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=max_lr)
trainable_parameters_num = sum(p.numel() for p in filter(lambda p:p.requires_grad, model.parameters()))  ##全参微调
```

配置logging日志记录模型训练过程

```python
##配置logging
import time

with open(f"./Qwen2.5-0.5B-DPO_log.txt", "a") as my_file:
    my_file.write(f' \
        time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, \
        batch_size:{batch_size}, \
        trainable_parameters_num:{trainable_parameters_num}, \
        warmup_steps:{warmup_steps}, \
        max_lr:{max_lr}, \
        min_lr:{min_lr}\n')

#定义一个日志记录函数
def log_call(iters, iters_average_loss):
    with open(f"./Qwen2.5-0.5B-DPO_log.txt", "a") as my_file:
        my_file.write(f' \
            time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, \
            iters:{iters+1}, \
            iters_average_Loss:{iters_average_loss:.4f}\n')
```

复用SFT阶段设置的余弦衰减学习率曲线

```python
def linear_warmup(current_step, warmup_steps, max_lr):
    if current_step < warmup_steps:
        return max_lr * current_step / warmup_steps
    else:
        return max_lr

def cosine_decay(current_step, warmup_steps, total_steps, max_lr, min_lr):
    if current_step < warmup_steps:
        return linear_warmup(current_step, warmup_steps, max_lr)
    else:
        progress = (current_step - warmup_steps) \
                 / (total_steps - warmup_steps)
        decay = 0.5 * (1 + np.cos(np.pi * progress))
        return (max_lr - min_lr) * decay + min_lr
```

复用SFT阶段设置的掩码设置

```python
def create_answer_mask(input_ids, tokenizer):
    """
    创建仅对助手回答部分计算损失的掩码
    
    Args:
        input_ids: 输入token序列 [batch_size, seq_len]
        tokenizer: 分词器
    
    Returns:
        answer_mask: 助手回答部分为1，其他部分为0的掩码
    """
    batch_size, seq_len = input_ids.shape
    answer_mask = torch.zeros_like(input_ids)
    
    # 获取<im_end>标记的token_id
    eos_token_id = tokenizer.encode('<|im_end|>')[0]
    
    for batch_idx in range(batch_size):
        # 找到所有 <|im_end|> 的位置
        eos_positions = torch.where(
            input_ids[batch_idx] == eos_token_id
        )[0].tolist()
        
        if len(eos_positions) < 2:  # 至少需要user和assistant各一个结束标记
            continue
            
        # 解析对话轮次
        user_ends, assistant_ends = _parse_conversation_turns(eos_positions)
        
        # 为每个助手回答设置掩码
        _set_answer_masks(
            answer_mask[batch_idx],
            user_ends,
            assistant_ends,
            seq_len
        )
    
    return answer_mask


def _parse_conversation_turns(eos_positions):
    """
    解析对话轮次，分离用户和助手的结束位置
    
    对话格式：
    <|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>\n
    
    eos_positions[0]: system结束 (如果有)
    eos_positions[1]: 第1轮user结束  
    eos_positions[2]: 第1轮assistant结束
    eos_positions[3]: 第2轮user结束
    eos_positions[4]: 第2轮assistant结束
    ...
    """
    # 跳过system部分，从第一个user开始
    conversation_eos = eos_positions[1:]  # 去掉system的<im_end>
    
    # 偶数索引：user结束位置，奇数索引：assistant结束位置
    user_ends = [pos + 1 for pos in conversation_eos[::2]] # 每隔2个取一个，从0开始
    assistant_ends = [pos + 1 for pos in conversation_eos[1::2]] # 每隔2个取一个，从1开始
    
    return user_ends, assistant_ends


def _set_answer_masks(mask, user_ends, assistant_ends, seq_len):
    """
    为助手回答部分设置掩码
    
    Args:
        mask: 当前样本的掩码 [seq_len]
        user_ends: 用户消息结束位置列表
        assistant_ends: 助手消息结束位置列表  
        seq_len: 序列长度
    """
    num_user_turns = len(user_ends)
    num_assistant_turns = len(assistant_ends)
    
    if num_user_turns == num_assistant_turns:
        # 完整对话：每轮都有用户问题和助手回答
        for user_end, assistant_end in zip(user_ends, assistant_ends):
            answer_start = user_end + 3  # 跳过 <|im_start|>assistant\n
            answer_end = assistant_end - 1  # 不包含 <|im_end|>
            mask[answer_start:answer_end] = 1
            
    elif num_user_turns == num_assistant_turns + 1:
        # 未完成对话：最后一轮助手回答被截断
        
        # 处理完整的对话轮次
        for user_end, assistant_end in zip(user_ends[:-1], assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1
        
        # 处理最后一轮被截断的助手回答
        last_user_end = user_ends[-1]
        last_answer_start = last_user_end + 3
        mask[last_answer_start:] = 1  # 到序列结尾
```

开启DPO训练

计算优势的辅助函数如下

```python
def _compute_average_log_probability(logits, target_labels, mask):
    """
    计算带掩码的平均对数概率
    
    Args:
        logits: 模型输出 [batch_size, seq_len, vocab_size]
        target_labels: 目标标签 [batch_size, seq_len]
        mask: 计算掩码 [batch_size, seq_len]
    
    Returns:
        average_log_prob: 每个样本的平均对数概率 [batch_size]
    """
    # 计算softmax概率分布
    probabilities = torch.softmax(logits, dim=-1)
    
    # 计算对数概率
    log_probabilities = torch.log(probabilities)
    
    # 获取目标token的对数概率
    gathered_log_probs = torch.gather(
        log_probabilities, 
        dim=-1, 
        index=target_labels.unsqueeze(2)
    ).squeeze(2)
    
    # 应用掩码并计算平均值
    masked_log_probs = torch.mul(gathered_log_probs, mask)
    average_log_prob = masked_log_probs.sum(dim=-1) / mask.sum(dim=-1)
    
    return average_log_prob
```

训练循环如下：

```python
model.train()

# ==================== 训练指标记录列表 ====================
training_losses = []
# 偏好的回答的概率
preferred_log_probabilities = []
# 讨厌的回答的概率
rejected_log_probabilities = []
# 偏好的回答的奖励
preferred_rewards = []
# 讨厌的回答的奖励
rejected_rewards = []
reward_margins = []

model.zero_grad()  # 训练开始时清空梯度
skipped_batches_count = 0
total_batches = len(chosen_input_ids_list) // batch_size

for batch_idx in range(total_batches):
    ## ==================== 获取批次数据 ====================
    
    # 获取当前批次的偏好对数据
    preferred_batch_sequences = chosen_input_ids_list[
        batch_idx * batch_size:(batch_idx + 1) * batch_size
    ]
    rejected_batch_sequences = rejected_input_ids_list[
        batch_idx * batch_size:(batch_idx + 1) * batch_size
    ]

    ## ==================== 数据填充对齐 ====================
    
    # 计算各自批次的最大序列长度
    preferred_max_length = max([len(sequence) for sequence in preferred_batch_sequences])
    rejected_max_length = max([len(sequence) for sequence in rejected_batch_sequences])
    # 使用eos token作为pad token
    pad_token_id = model.generation_config.eos_token_id[-1]
    
    ### 偏好数据填充处理
    preferred_padded_sequences = []
    for seq_idx in range(batch_size):
        original_sequence = preferred_batch_sequences[seq_idx]
        # 计算要填充多少个pad
        padding_length = preferred_max_length - len(original_sequence)
        # 在训练数据的末尾填充pad
        padded_sequence = torch.nn.functional.pad(
            torch.tensor(original_sequence), 
            (0, padding_length), 
            mode='constant', 
            value=pad_token_id
        ).tolist()
        # 将填充过的数据放入列表
        preferred_padded_sequences.append(padded_sequence)
    
    preferred_batch_tensor = torch.tensor(preferred_padded_sequences)
    
    ### 拒绝数据填充处理
    rejected_padded_sequences = []
    for seq_idx in range(batch_size):
        original_sequence = rejected_batch_sequences[seq_idx]
        padding_length = rejected_max_length - len(original_sequence)
        
        padded_sequence = torch.nn.functional.pad(
            torch.tensor(original_sequence), 
            (0, padding_length), 
            mode='constant', 
            value=pad_token_id
        ).tolist()
        
        rejected_padded_sequences.append(padded_sequence)
    
    rejected_batch_tensor = torch.tensor(rejected_padded_sequences)

    ## ==================== 构建输入输出对 ====================
    
    # 构建因果语言模型的输入输出对：x->y（下一个词预测）
    # 模型的输入：偏好的回答
    preferred_model_inputs = preferred_batch_tensor[:, :-1].to(device)
    # 真实的标签
    preferred_target_labels = preferred_batch_tensor[:, 1:].to(device)
    
    rejected_model_inputs = rejected_batch_tensor[:, :-1].to(device)
    rejected_target_labels = rejected_batch_tensor[:, 1:].to(device)

    ## ==================== 构建训练掩码 ====================
    
    # 构建掩码矩阵：padding_mask（忽略填充token）+ answer_mask（只关注回答部分）
    
    # pad_token_id 对应的置为 0 ，其它置为 1 。
    preferred_padding_mask = torch.where(
        preferred_target_labels == pad_token_id,
        0,
        1
    )
    rejected_padding_mask = torch.where(
        rejected_target_labels == pad_token_id,
        0,
        1
    )
    
    # 助手回答的掩码：将助手回答的部分掩码为 1 。其它都是 0 。
    preferred_answer_mask = create_answer_mask(
        preferred_model_inputs,
        tokenizer
    )
    rejected_answer_mask = create_answer_mask(
        rejected_model_inputs,
        tokenizer
    )
    
    # 最终掩码：取交集
    preferred_final_mask = (preferred_answer_mask & preferred_padding_mask)
    rejected_final_mask = (rejected_answer_mask & rejected_padding_mask)

    ## ==================== 批次有效性检查 ====================
    
    # 检查偏好对数据是否都有有效的回答部分
    preferred_min_tokens = preferred_final_mask.sum(dim=-1).min().item()
    rejected_min_tokens = rejected_final_mask.sum(dim=-1).min().item()
    
    if preferred_min_tokens == 0 or rejected_min_tokens == 0:
        print(f'⚠️ 跳过第{batch_idx + 1}批次：偏好对数据回答部分不足')
        skipped_batches_count += 1
        continue  # 跳过当前批次

    ## ==================== 模型前向传播 ====================
    
    # 训练模型对偏好数据的前向传播
    preferred_logits = model(preferred_model_inputs).logits
    torch.cuda.empty_cache()  # 清理GPU显存
    torch.cuda.ipc_collect()

    # 训练模型对拒绝数据的前向传播
    rejected_logits = model(rejected_model_inputs).logits
    torch.cuda.empty_cache()  # 清理GPU显存
    torch.cuda.ipc_collect()

    # 参考模型的前向传播（不计算梯度）
    with torch.no_grad():
        reference_preferred_logits = ref_model(preferred_model_inputs) \
            .logits                                                    \
            .detach()
        reference_rejected_logits = ref_model(rejected_model_inputs)   \
            .logits                                                    \
            .detach()

    ## ==================== DPO损失计算 ====================
    """
    DPO (Direct Preference Optimization) 论文: https://arxiv.org/pdf/2305.18290.pdf
    核心思想：通过偏好对比学习，无需显式奖励模型
    """
    
    # 计算平均对数概率 (average_log_prob = True)
    # 参考: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L924
    
    ### 训练模型的对数概率
    ### 正在微调的模型，接收到正例的logits，计算对数概率
    preferred_log_prob = _compute_average_log_probability(
        preferred_logits,
        preferred_target_labels,
        preferred_final_mask
    )
    rejected_log_prob = _compute_average_log_probability(
        rejected_logits,
        rejected_target_labels,
        rejected_final_mask
    )
    
    ### 参考模型的对数概率
    reference_preferred_log_prob = _compute_average_log_probability(
        reference_preferred_logits,
        preferred_target_labels,
        preferred_final_mask
    )
    reference_rejected_log_prob = _compute_average_log_probability(
        reference_rejected_logits,
        rejected_target_labels,
        rejected_final_mask
    )

    ## ==================== 奖励和边际计算 ====================
    
    # 计算隐式奖励 (基于KL散度)
    preferred_implicit_reward =                              \
        beta *                                               \
        (preferred_log_prob - reference_preferred_log_prob)
    rejected_implicit_reward =                               \
        beta *                                               \
        (rejected_log_prob - reference_rejected_log_prob)
    
    # 计算奖励边际 (偏好数据应该有更高的奖励)
    reward_margin = preferred_implicit_reward - rejected_implicit_reward
    
    # DPO损失：-log(sigmoid(margin))
    preference_probability = torch.nn.functional.sigmoid(reward_margin)
    sample_losses = -torch.log(preference_probability)
    
    # 批次平均损失 + 梯度累积
    batch_average_loss =                          \
        torch.nanmean(sample_losses) /            \
        gradient_accumulation_steps

    ## ==================== 反向传播和优化 ====================
    
    batch_average_loss.backward()

    # 动态学习率调整
    current_learning_rate = cosine_decay(
        batch_idx,
        warmup_steps,
        total_steps,
        max_lr,
        min_lr
    )
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_learning_rate

    # 梯度累积和权重更新
    is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
    is_final_batch = (batch_idx + 1) == total_batches
    
    if is_accumulation_step or is_final_batch:
        optimizer.step()        # 更新权重
        optimizer.zero_grad()   # 清空梯度

    ## ==================== 训练指标记录 ====================
    
    # 记录各项训练指标（detach避免梯度追踪）
    training_losses.append(
        batch_average_loss.detach().item() * gradient_accumulation_steps)
    preferred_log_probabilities.append(
        torch.nanmean(preferred_log_prob.detach()).item())
    rejected_log_probabilities.append(
        torch.nanmean(rejected_log_prob.detach()).item())
    preferred_rewards.append(
        torch.nanmean(preferred_implicit_reward.detach()).item())
    rejected_rewards.append(torch.nanmean(
        rejected_implicit_reward.detach()).item())
    reward_margins.append(
        torch.nanmean(reward_margin.detach()).item())

    ## ==================== 训练日志输出 ====================
    
    should_log = (batch_idx + 1) % log_iter == 0 or is_final_batch
    
    if should_log:
        # 计算最近批次的平均指标
        recent_loss = np.nanmean(training_losses[-log_iter:])
        recent_preferred_logprob = np.nanmean(
            preferred_log_probabilities[-log_iter:])
        recent_rejected_logprob = np.nanmean(
            rejected_log_probabilities[-log_iter:])
        recent_preferred_reward = np.nanmean(preferred_rewards[-log_iter:])
        recent_rejected_reward = np.nanmean(rejected_rewards[-log_iter:])
        recent_margin = np.nanmean(reward_margins[-log_iter:])
        
        # 格式化输出训练状态
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f'⏰ 时间: {current_time}')
        print(f'📊 批次: {batch_idx + 1}/{total_batches}')
        print(f'📈 最近{log_iter}批次指标:')
        print(f'   - 平均损失: {recent_loss:.4f}')
        print(f'   - 偏好对数概率: {recent_preferred_logprob:.4f}')
        print(f'   - 拒绝对数概率: {recent_rejected_logprob:.4f}')
        print(f'   - 偏好奖励: {recent_preferred_reward:.4f}')
        print(f'   - 拒绝奖励: {recent_rejected_reward:.4f}')
        print(f'   - 奖励边际: {recent_margin:.4f}')
        print(f'🎯 学习率: {current_learning_rate:.2e}')
        print('-' * 80)
        
        # 调用外部日志记录
        log_call(batch_idx, recent_loss)

## ==================== 训练完成总结 ====================

print("🎉 DPO训练完成!")
print(f'📊 训练统计:')
print(f'   - 总批次数: {total_batches}')
print(f'   - 跳过批次数: {skipped_batches_count}')
print(f'   - 有效批次数: {total_batches - skipped_batches_count}')

# 输出最终训练指标
if training_losses:
    final_metrics = {
        'loss': np.nanmean(training_losses[-100:]),
        'preferred_logprob': np.nanmean(preferred_log_probabilities[-100:]),
        'rejected_logprob': np.nanmean(rejected_log_probabilities[-100:]),
        'preferred_reward': np.nanmean(preferred_rewards[-100:]),
        'rejected_reward': np.nanmean(rejected_rewards[-100:]),
        'margin': np.nanmean(reward_margins[-100:])
    }
    
    print(f'🎯 最终指标 (最近100批次平均):')
    for metric_name, metric_value in final_metrics.items():
        print(f'   - {metric_name}: {metric_value:.4f}')

if skipped_batches_count > 0:
    skip_ratio = skipped_batches_count / total_batches * 100
    print(f'⚠️ 跳过批次占比: {skip_ratio:.2f}%')
    if skip_ratio > 10:
        print('💡 建议: 跳过批次过多，考虑增加最大序列长度或优化数据预处理')
```

保存模型

```python
model.save_pretrained('./Qwen2.5-0.5B-DPO')
tokenizer.save_pretrained('./Qwen2.5-0.5B-DPO')
```

## 4. 使用推理脚本测试

对预训练模型 `Qwen2.5-0.5B` 测试

对SFT后的模型 `Qwen2.5-0.5B-SFT` 测试

对DPO后的模型 `Qwen2.5-0.5B-DPO` 测试

对比以上结果。

## 5. 在Jupyter Notebook中的推理代码

```python
from IPython.display import Markdown, display

history = []
history.append({"role": "system", "content": "You are a helpful assistant"})
while True:
    question = input('User：' + '\n')
    print('\n')
    history.append({"role": "user", "content": question})

    input_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )
    model_inputs = tokenizer([input_text], return_tensors="pt").to(device)

    if model_inputs.input_ids.size()[1]>32000:
        break

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1000
    )

    if len(generated_ids)>32000:
        break

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('Assistant:\n')
    #print(response)
    display(Markdown(response))
    print("--------------------")
    print('\n')
    history.append({"role": "assistant", "content": response})

print("超过模型字数上线，已退出")
```

