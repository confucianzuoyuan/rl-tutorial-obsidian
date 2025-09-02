from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW


device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "../Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("../Qwen2.5-0.5B-Instruct")


def chat(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # print(text)

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    return response


prompt = "你是谁？"
response = chat(prompt)
print(response)


def preprocess(tokenizer, batch_messages):
    '''训练数据预处理方法'''
    input_list = []
    target_list = []

    im_start = tokenizer('<|im_start|>').input_ids
    im_end = tokenizer('<|im_end|>').input_ids
    newline = tokenizer('\n').input_ids
    pad = tokenizer('<|endoftext|>').input_ids
    ignore = [-100]

    for group in batch_messages:
        input_ids = []
        target_ids = []
        for msg in group:
            role = tokenizer(msg['role']).input_ids
            content = tokenizer(msg['content']).input_ids
            if msg['role'] in ['system', 'user']:
                ignore_parts = role+newline+content
                input_ids += im_start+ignore_parts+im_end+newline
                target_ids += im_start+ignore*len(ignore_parts)+im_end+newline
            else:
                ignore_parts = role+newline
                input_ids += im_start+ignore_parts+content+im_end+newline
                target_ids += im_start+ignore * \
                    len(ignore_parts)+content+im_end+newline
        input_list.append(input_ids)
        target_list.append(target_ids)

    # padding
    max_len = max([len(ids) for ids in input_list])
    for input_ids, target_ids in zip(input_list, target_list):
        input_ids += pad*(max_len-len(input_ids))
        target_ids += ignore*(max_len-len(target_ids))
    batch_input_ids = torch.tensor(input_list, dtype=torch.long)
    batch_target_ids = torch.tensor(target_list, dtype=torch.long)
    batch_mask = batch_input_ids.ne(pad[0]).type(torch.long)
    return batch_input_ids, batch_target_ids, batch_mask


prompt = "你是谁发明的?"
messages = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": '尚硅谷大模型研发组'},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": '尚硅谷大模型研发组'},
    ]
]

model.train()

optimizer = AdamW(model.parameters(), lr=1e-4)


for i in range(10):
    batch_input_ids, batch_target_ids, batch_mask = preprocess(
        tokenizer, messages)
    model_outputs = model(batch_input_ids.to(device))

    output_tokens = model_outputs.logits.argmax(dim=-1)

    logits = model_outputs.logits[:, :-1, :]
    targets = batch_target_ids[:, 1:].to(device)

    # 损失
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits.reshape(-1, logits.size(2)), targets.reshape(-1))
    print('步骤：', i, '，损失：', loss.item())

    # 优化器
    optimizer.zero_grad()

    # 求梯度
    loss.backward()

    # 梯度下降
    optimizer.step()

model.eval()
print('回答：', chat('你是谁发明的?'))
