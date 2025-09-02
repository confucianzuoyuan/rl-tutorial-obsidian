import torch
from model import g, device, pad_id, eos_id
from transformers import GPT2LMHeadModel
import torch.nn.functional as F


@torch.no_grad()
def get_question_and_answer():
    _, token, _ = g.get_batch_data(prefix=True)

    split = [i.index(g.encoder['=']) + 1 for i in token]

    # 只要问题部分,等号后面的内容切除
    question = [t[:s] for t, s in zip(token, split)]
    answer = [t[s:] for t, s in zip(token, split)]

    # 统一长度
    lens = max([len(i) for i in question])
    question = [[pad_id] * (lens - len(i)) + i
                for i in question]
    question = torch.LongTensor(question).to(device)

    lens = max([len(i) for i in answer])
    answer = [[pad_id] * (lens - len(i)) + i for i in answer]
    answer = torch.LongTensor(answer).to(device)

    return question, answer


@torch.no_grad()
def get_question():
    label, question, _ = g.get_batch_data(prefix=True)
    label = torch.LongTensor(label).to(device)

    # 只要问题部分,等号后面的内容切除
    question = [i[:i.index(g.encoder['=']) + 1] for i in question]

    # 统一长度
    max_length = max([len(i) for i in question])
    question = [[g.encoder['P']] * (max_length - len(i)) + i
                for i in question]

    question = torch.LongTensor(question).to(device)

    return label, question

# 如果question的长度确定,这里可以转换成批运算


@torch.no_grad()
def get_answer(model, question):
    answer = response_generate(model, question)

    # 裁剪,只要生成的部分
    answer = answer[:, question.shape[1]:]

    return answer


@torch.no_grad()
def get_reward(reward_model, question, answer, label):
    input_ids = torch.cat((question, answer), 1)
    attention_mask = (input_ids != pad_id).long()

    with torch.no_grad():
        logits = reward_model(input_ids=input_ids,
                              attention_mask=attention_mask)

    return logits.gather(1, label.reshape(-1, 1)).squeeze(1)


@torch.no_grad()
def response_generate(llm_model, input_ids):
    response_generator = GPT2LMHeadModel(llm_model.config)
    response_generator.transformer = llm_model.feature
    response_generator.lm_head = llm_model.fc_out
    response_generator.to(device)
    res = response_generator.generate(
        input_ids=input_ids,
        min_length=-1,
        top_k=0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=pad_id,
        max_new_tokens=25,
        eos_token_id=eos_id
    )
    return res


def logprobs_from_logits(logits, labels, gather=True):
    """
    从模型输出的 logits 计算对应的对数概率
    """
    # 对 logits 在最后一个维度做 log_softmax，得到每个类别的对数概率
    logp = F.log_softmax(logits, dim=2)

    # 如果不需要特定标签的概率，就直接返回所有类别的对数概率
    if not gather:
        return logp

    # labels 是目标类别索引，形状通常是 [batch_size, seq_len]
    # labels.unsqueeze(2) 形状变成 [batch_size, seq_len, 1]
    # torch.gather 从 logp 中按标签索引取对应位置的 log 概率，结果形状是 [batch_size, seq_len, 1]
    # 去掉最后一个维度，变成 [batch_size, seq_len]
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)

    # 返回对应标签的对数概率
    return logpy


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened
