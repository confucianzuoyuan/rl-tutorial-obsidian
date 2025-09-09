import time
import datasets
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import os

device = "cuda"  # the device to load the model onto
model_name = "Qwen2.5-0.5B"
model_path = "../../Qwen2.5-0.5B"

# 获取可用GPU设备的数量和内存信息
gpu_memory = {i: torch.cuda.get_device_properties(
    i).total_memory for i in range(torch.cuda.device_count())}
# 加载模型
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="balanced",
        max_memory={
            i: f"{gpu_memory[i] // (1024 ** 3)}GB" for i in range(len(gpu_memory))}
    )
except Exception as e:
    print(f"Failed to load model from {model_path}: {e}")
    raise
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 重设模型的generation_config文件，以便最后对比训练前后生成式问答的效果
print(model.generation_config)

model.generation_config.do_sample = True
model.generation_config.eos_token_id = [151645, 151643]
model.generation_config.pad_token_id = 151643
model.generation_config.temperature = 0.7
model.generation_config.top_p = 0.8
model.generation_config.top_k = 20
model.generation_config.repetition_penalty = 1.05

print(model.generation_config)


@dataclass
class SFTConfig:
    max_length: int = 2500
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    log_iter: int = 400
    max_lr: float = 2e-5
    min_lr: float = 2e-6
    warmup_steps: int = 1000


ultrachat_200k_data = datasets.load_dataset('../../ultrachat_200k')

# print(ultrachat_200k_data["train_gen"][0].keys())


def tokenize_and_format(data):  # data: List[Dict[str, str]]
    input_ids = tokenizer.apply_chat_template(
        data,  # Union[List[Dict[str, str]]
        tokenize=True,
        add_generation_prompt=False,
        # padding = True,
        truncation=True,
        max_length=SFTConfig.max_length,
    )

    return input_ids  # tag_ids


# 生成训练数据的tokenid
chosen_input_ids_list = []
i = 0
while True:
    data = ultrachat_200k_data['train_sft'][i]['messages']
    data.insert(
        0, {"content": "You are a helpful assistant", "role": "system"})
    input_ids = tokenize_and_format(data)
    chosen_input_ids_list.append(input_ids)
    i += 1
    if i % 1000 == 0:
        print(f"已处理{i}条数据")
    if i == 50000:  # len(ultrachat_200k_data['train_sft']):
        break

# 使用设置的训练超参数

batch_size = SFTConfig.batch_size
gradient_accumulation_steps = SFTConfig.gradient_accumulation_steps
log_iter = SFTConfig.log_iter
max_lr = SFTConfig.max_lr
min_lr = SFTConfig.min_lr
warmup_steps = SFTConfig.warmup_steps
total_steps = len(chosen_input_ids_list)//batch_size
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=max_lr)
trainable_parameters_num = sum(p.numel() for p in filter(
    lambda p: p.requires_grad, model.parameters()))  # 全参微调


# 配置logging日志记录模型训练过程

# 配置logging
with open(f"./{model_name}-SFT_log.txt", "a") as my_file:
    my_file.write(f'time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, batch_size:{batch_size}, trainable_parameters_num:{trainable_parameters_num}, warmup_steps:{warmup_steps}, max_lr:{max_lr}, min_lr:{min_lr}\n')
# 定义一个日志记录函数


def log_call(iters, iters_average_loss):
    with open(f"./{model_name}-SFT_log.txt", "a") as my_file:
        my_file.write(
            f'time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, iters:{iters+1}, iters_average_Loss:{iters_average_loss:.4f}\n')

# 学习率设置：余弦衰减学习率


def linear_warmup(current_step, warmup_steps, max_lr):
    if current_step < warmup_steps:
        return max_lr * current_step / warmup_steps
    else:
        return max_lr


def cosine_decay(current_step, warmup_steps, total_steps, max_lr, min_lr):
    if current_step < warmup_steps:
        return linear_warmup(current_step, warmup_steps, max_lr)
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        decay = 0.5 * (1 + np.cos(np.pi * progress))
        return (max_lr - min_lr) * decay + min_lr

# 掩码设置

# - SFT和预训练的区别核心就是掩码掉“问题”部分的损失，而【只看“回答”部分的损失】，并仅基于回答部分的损失进行优化
# - 实现方式：【构造损失掩码】，仅针对每轮对话（含多轮）的模型“输出”部分（也就是回答部分）进行损失计算

# 设置问题部分的掩码函数，用于执行仅针对回答部分才计算损失


def return_answer_mask(input_ids):
    assistant_answer_mask = torch.zeros_like(input_ids)  # 0初始化
    for i in range(input_ids.shape[0]):
        # user部分的结尾\n: \n是<|im_end|>的下一个元素，所以有+1 【这个地方需要根据不同模型的不同聊天模版自定义更改】，关于聊天模版可阅读这篇文章：https://huggingface.co/blog/chat-templates
        i_user_end_list = [
            i+1 for i in torch.where(input_ids[i] == tokenizer.encode('<|im_end|>')[0])[0].tolist()[1::2]]
        # assistant部分的结尾\n：\n是<|im_end|>的下一个元素，所以有+1 【这个地方需要根据不同模型的不同聊天模版自定义更改】
        i_assistant_end_list = [
            i+1 for i in torch.where(input_ids[i] == tokenizer.encode('<|im_end|>')[0])[0].tolist()[2::2]]

        if len(i_user_end_list) == len(i_assistant_end_list):
            for user_end, assistant_end in zip(i_user_end_list, i_assistant_end_list):
                # +3的操作，【这个地方需要根据不同模型的不同聊天模版自定义更改】
                assistant_answer_mask[i][user_end+3:assistant_end-1] = 1
        elif len(i_user_end_list) == len(i_assistant_end_list)+1 == 1:  # 单轮问答,且回答部分未结尾就被截断了
            # 会把右补的padding token也标记为1，所以后面还需要再结合padding mask以过滤padding
            assistant_answer_mask[i][i_user_end_list[0]+3:] = 1
        elif len(i_user_end_list) == len(i_assistant_end_list)+1:  # 兼顾多轮问答
            assistant_answer_mask[i][i_user_end_list[-1]+3:] = 1
            for user_end, assistant_end in zip(i_user_end_list[:-1], i_assistant_end_list):
                assistant_answer_mask[i][user_end+3:assistant_end-1] = 1
        else:
            continue  # 跳出当前循环，继续下一次循环
    return assistant_answer_mask

# 开启SFT微调训练


model.train()
train_loss_list = []
model.zero_grad()  # clear gradients at the start of training
ignore_iters_count = 0
for iters in range(len(chosen_input_ids_list)//batch_size):
    # 获取批次数据
    chosen_batch_inputids = chosen_input_ids_list[iters *
                                                  batch_size:(iters+1)*batch_size]

    # 对该批次数据进行padding,以并行计算，首先计算该批次的最大token长度
    chosen_max_dim = max([len(i) for i in chosen_batch_inputids])

    # 训练数据padding填充
    chosen_batch_inputids_padding_list = []
    for i in range(batch_size):
        chosen_batch_inputids_padding_list.append(torch.nn.functional.pad(torch.tensor(chosen_batch_inputids[i]), (
            # 右补
            0, chosen_max_dim - len(chosen_batch_inputids[i])), mode='constant', value=model.generation_config.eos_token_id[-1]).tolist())
    chosen_batch_inputids_tensor = torch.tensor(
        chosen_batch_inputids_padding_list)

    # 构建训练数据：x->y ,下一个单词预测
    chosen_x = chosen_batch_inputids_tensor[:, :-1].to(device)
    chosen_y = chosen_batch_inputids_tensor[:, 1:].to(device)

    # 构建掩码判别矩阵（paddding mask & answer_mask, padding mask用于执行对padding的token不计算损失，answer_mask用于执行仅针对回答部分才计算损失），总之，就是确认哪些tokens的logit需要"忽视"掉
    # 【padding mask】
    chosen_padding_mask = torch.where(
        chosen_y == model.generation_config.eos_token_id[-1], 0, 1)
    # 【answer_mask】
    chosen_assistant_answer_mask = return_answer_mask(chosen_x)
    # 【paddingmask & answermask】方便使用掩码判别矩阵对logit和y进行过滤->:我们只关注【回答】部分的损失，不关注问题部分的损失
    chosen_assistant_answer_mask = (
        chosen_assistant_answer_mask & chosen_padding_mask)

    # 如果该批次里有的问答数据在数据截取时，回答部分存在没有数据的情况（问题太长了，导致还未采集到回答部分的token就被硬截断了），那么该批次数据不再训练
    if chosen_assistant_answer_mask.sum(dim=-1).min().item() == 0:
        # print(f'不处理第{iters+1}批次数据')
        ignore_iters_count += 1
        continue  # 跳出当前循环

    # 执行训练数据的模型前向推理，计算logits
    chosen_logits = model(chosen_x).logits
    torch.cuda.empty_cache()  # 清除非必要的显存占用，但会导致速度变慢
    torch.cuda.ipc_collect()

    # Compute Chosen_Answer_Loss，计算训练数据的回答部分的损失, batch_loss的shape_size:[batch_size]
    batch_loss = torch.mul((torch.gather(torch.log(torch.softmax(chosen_logits, dim=-1)), dim=-1, index=chosen_y.unsqueeze(2))
                           * (-1)).squeeze(2), chosen_assistant_answer_mask).sum(dim=-1) / chosen_assistant_answer_mask.sum(dim=-1)

    # Calculate the Final Loss, 只是新增了梯度积累的操作
    loss = torch.nanmean(batch_loss)/(gradient_accumulation_steps)

    loss.backward()  # 反向传播计算梯度

    # Compute the learning rate for the current step
    lr = cosine_decay(iters, warmup_steps, total_steps, max_lr, min_lr)

    # Update the learning rate for the AdamW optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if (iters+1) % gradient_accumulation_steps == 0 or (iters+1) == (len(chosen_input_ids_list)//batch_size):
        optimizer.step()  # update weights after gradients accumulation
        # at last, clear gradients
        # clear gradients after updating, in this case equal to model.zero_grad()
        optimizer.zero_grad()

    train_loss_list.append(loss.item()*gradient_accumulation_steps)

    if (iters+1) % log_iter == 0 or (iters+1) == (len(chosen_input_ids_list)//batch_size):
        # 避免空值影响
        print(
            f'time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, iters:{iters+1}, last_{log_iter}_iters_average_train_Loss:{np.nanmean(train_loss_list[-log_iter:]):.4f}')
        log_call(iters, np.nanmean(train_loss_list[-log_iter:]))

print("Totally Completed!")
print(f'共计忽略{ignore_iters_count}个批次数据')
model.save_pretrained("./Qwen2.5-0.5B-SFT/")
tokenizer.save_pretrained("./Qwen2.5-0.5B-SFT/")
