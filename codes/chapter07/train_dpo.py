# train_dpo.py
import pprint
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import copy

device = "cuda"

def chat(prompt, tokenizer, model):
    '''测试函数'''
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        [text], 
        return_tensors="pt", 
        padding=True,
        truncation=True
    ).to(device)

    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    return response

class dpo_dataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # 打开json文件 用transformers
        self.data_list = load_dataset("json", data_files=file)['train']

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 取出data_list的一条数据  --> {"chosen":xxx,"rejected":xxx,"prompt":xxx} 一条数据是这样的格式
        data = self.data_list[index]

        # 对prompt reject和chosen进行tokenize  判断是否需要截断 保证所有的input_ids都一样 不够长度的直接padding
        # 适配qwen 的 template  添加eos token
        prompt_input_ids = self.tokenizer.encode(
            '<|im_start|>' + data['prompt'] + '<|im_end|>', add_special_tokens=False)
        chosen_input_ids = self.tokenizer.encode(
            data['chosen'], add_special_tokens=False)
        rejected_input_ids = self.tokenizer.encode(
            data['rejected'], add_special_tokens=False)

        prompt_input_ids = prompt_input_ids + [self.tokenizer.pad_token_id]
        # 设置labels
        chosen_labels = [-100] * len(prompt_input_ids) + \
            chosen_input_ids + [self.tokenizer.pad_token_id]
        rejected_labels = [-100] * len(prompt_input_ids) + \
            rejected_input_ids + [self.tokenizer.pad_token_id]
        chosen_input_ids = prompt_input_ids + \
            chosen_input_ids + [self.tokenizer.pad_token_id]
        rejected_input_ids = prompt_input_ids + \
            rejected_input_ids + [self.tokenizer.pad_token_id]

        assert len(chosen_labels) == len(chosen_input_ids)
        assert len(rejected_labels) == len(rejected_input_ids)

        inputs = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=[1]*len(prompt_input_ids),
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=[1]*len(chosen_input_ids),
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=[1]*len(rejected_input_ids),
            rejected_labels=rejected_labels,
        )
        return inputs

    def map(self, func, **kwargs):
        return self

def collate_fn(batch):
    """数据整理函数，处理批次数据的padding"""
    # 找到最大长度
    max_chosen_len = max(len(item['chosen_input_ids']) for item in batch)
    max_rejected_len = max(len(item['rejected_input_ids']) for item in batch)
    max_len = max(max_chosen_len, max_rejected_len)
    
    # 准备批次数据
    batch_data = {
        'chosen_input_ids': [],
        'chosen_attention_mask': [],
        'chosen_labels': [],
        'rejected_input_ids': [],
        'rejected_attention_mask': [],
        'rejected_labels': []
    }
    
    for item in batch:
        # Padding chosen
        chosen_len = len(item['chosen_input_ids'])
        pad_len = max_len - chosen_len
        
        batch_data['chosen_input_ids'].append(
            item['chosen_input_ids'] + [tokenizer.pad_token_id] * pad_len
        )
        batch_data['chosen_attention_mask'].append(
            item['chosen_attention_mask'] + [0] * pad_len
        )
        batch_data['chosen_labels'].append(
            item['chosen_labels'] + [-100] * pad_len
        )
        
        # Padding rejected
        rejected_len = len(item['rejected_input_ids'])
        pad_len = max_len - rejected_len
        
        batch_data['rejected_input_ids'].append(
            item['rejected_input_ids'] + [tokenizer.pad_token_id] * pad_len
        )
        batch_data['rejected_attention_mask'].append(
            item['rejected_attention_mask'] + [0] * pad_len
        )
        batch_data['rejected_labels'].append(
            item['rejected_labels'] + [-100] * pad_len
        )
    
    # 转换为tensor
    for key in batch_data:
        batch_data[key] = torch.tensor(batch_data[key], dtype=torch.long)
    
    return batch_data

def compute_sequence_log_prob(labels, logits):
    """计算序列的对数概率"""
    # 移位处理
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # 计算log概率
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # 创建mask，忽略-100的位置
    mask = (shift_labels != -100).float()
    
    # 收集目标token的概率
    gathered_log_probs = torch.gather(
        log_probs, 
        dim=-1, 
        index=shift_labels.unsqueeze(-1).clamp(min=0)
    ).squeeze(-1)
    
    # 应用mask并计算平均概率
    masked_log_probs = gathered_log_probs * mask
    sequence_log_prob = masked_log_probs.sum(-1) / (mask.sum(-1) + 1e-8)
    
    return sequence_log_prob

def dpo_loss(pi_chosen_logits, pi_rejected_logits, ref_chosen_logits, ref_rejected_logits,
             chosen_labels, rejected_labels, beta=0.1):
    """DPO损失函数"""
    
    # 计算各个模型的序列概率
    pi_chosen_prob = compute_sequence_log_prob(chosen_labels, pi_chosen_logits)
    pi_rejected_prob = compute_sequence_log_prob(rejected_labels, pi_rejected_logits)
    ref_chosen_prob = compute_sequence_log_prob(chosen_labels, ref_chosen_logits)
    ref_rejected_prob = compute_sequence_log_prob(rejected_labels, ref_rejected_logits)
    
    # 计算log ratios
    pi_logratios = pi_chosen_prob - pi_rejected_prob
    ref_logratios = ref_chosen_prob - ref_rejected_prob
    
    # DPO损失
    loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    
    return loss.mean()

class CustomDPOTrainer:
    def __init__(self, model, tokenizer, train_dataset, num_epochs=20, learning_rate=5e-6, beta=0.1):
        self.model_pi = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.beta = beta
        
        # 创建参考模型（冻结参数）
        self.model_ref = copy.deepcopy(model)
        self.model_ref.eval()
        for param in self.model_ref.parameters():
            param.requires_grad_(False)
        
        # 设置优化器
        self.optimizer = torch.optim.AdamW(self.model_pi.parameters(), lr=learning_rate)
        
        # 创建数据加载器
        self.dataloader = DataLoader(
            train_dataset, 
            batch_size=1,  # 根据显存调整
            shuffle=True, 
            collate_fn=collate_fn
        )
    
    def train(self):
        """训练函数"""
        self.model_pi.train()
        
        total_steps = len(self.dataloader) * self.num_epochs
        step = 0
        
        print(f"开始训练，总共 {self.num_epochs} 个epoch，{len(self.dataloader)} 个batch")
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(self.dataloader):
                # 将数据移到GPU
                for key in batch:
                    batch[key] = batch[key].to(device)
                
                # 前向传播 - 参考模型
                with torch.no_grad():
                    ref_chosen_outputs = self.model_ref(
                        input_ids=batch['chosen_input_ids'],
                        attention_mask=batch['chosen_attention_mask']
                    )
                    ref_rejected_outputs = self.model_ref(
                        input_ids=batch['rejected_input_ids'],
                        attention_mask=batch['rejected_attention_mask']
                    )
                
                # 前向传播 - 训练模型
                pi_chosen_outputs = self.model_pi(
                    input_ids=batch['chosen_input_ids'],
                    attention_mask=batch['chosen_attention_mask']
                )
                pi_rejected_outputs = self.model_pi(
                    input_ids=batch['rejected_input_ids'],
                    attention_mask=batch['rejected_attention_mask']
                )
                
                # 计算DPO损失
                loss = dpo_loss(
                    pi_chosen_outputs.logits,
                    pi_rejected_outputs.logits,
                    ref_chosen_outputs.logits,
                    ref_rejected_outputs.logits,
                    batch['chosen_labels'],
                    batch['rejected_labels'],
                    beta=self.beta
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model_pi.parameters(), max_norm=1.0)
                
                # 更新参数
                self.optimizer.step()
                
                epoch_loss += loss.item()
                step += 1
                
                # 打印进度
                if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], "
                          f"Batch [{batch_idx+1}/{len(self.dataloader)}], "
                          f"Loss: {loss.item():.6f}")
            
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] 完成, 平均损失: {avg_loss:.6f}")
            
            # 每几个epoch测试一次
            if (epoch + 1) % 5 == 0:
                self.model_pi.eval()
                test_output = chat("你们这有断桥铝窗吗?", self.tokenizer, self.model_pi)
                print(f"  测试输出: {test_output}")
                self.model_pi.train()
        
        print("训练完成!")

# 主程序
model = AutoModelForCausalLM.from_pretrained("./Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-0.5B-Instruct")

# 设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)

print("训练前测试:")
print("问题1:", chat("你们这有断桥铝窗吗?", tokenizer, model))
print("问题2:", chat("你们这里有氨氮去除菌剂和泄爆门吗?", tokenizer, model))
print()

# 创建数据集
train_dataset = dpo_dataset(file='dpo_data.json', tokenizer=tokenizer, max_seq_length=50)

# 创建自定义训练器
trainer = CustomDPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    num_epochs=20,
    learning_rate=5e-6,
    beta=0.1
)

# 开始训练
trainer.train()

print("\n训练后测试:")
print("问题1:", chat("你们这有断桥铝窗吗?", tokenizer, model))
print("问题2:", chat("你们这里有氨氮去除菌剂和泄爆门吗?", tokenizer, model))

# 保存模型
# model.save_pretrained("./Qwen2.5-0.5B-DPO-custom")
# tokenizer.save_pretrained("./Qwen2.5-0.5B-DPO-custom")
# print("\n模型已保存到 ./Qwen2.5-0.5B-DPO-custom")
