import torch
from model import LLM_Model, g, pad_id, device

# 初始化一个模型
llm_model = LLM_Model().to(device)
llm_model.train()
# 设置使用AdamW优化器
optimizer = torch.optim.AdamW(llm_model.parameters(), lr=1e-4)
# 评估标准，忽略掉 `P` 填充标记
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

# 训练 15000 轮
for epoch in range(15000):
    _, input_ids, attention_mask = g.get_batch_data(prefix=False)
    input_ids = torch.LongTensor(input_ids).to(device)
    attention_mask = torch.LongTensor(attention_mask).to(device)

    logits = llm_model(input_ids=input_ids, attention_mask=attention_mask)
    loss = criterion(logits[:, :-1].flatten(end_dim=1),
                     input_ids[:, 1:].flatten())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

llm_model.to('cpu')
torch.save(llm_model, 'llm.pt')
