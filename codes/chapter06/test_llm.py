import torch
from model import LLM_Model, device, g
from utils import response_generate

llm_model = LLM_Model()
llm_model = torch.load('llm.pt', weights_only=False)
llm_model.eval()  # 设置为评估模式


# 测试生成示例
_, input_ids, attention_mask = g.get_batch_data(prefix=True)
print(g.decode(input_ids[0][:11]))
print(g.decode(input_ids[1][:11]))
input_ids = torch.LongTensor(input_ids).to(device)

res = response_generate(llm_model, input_ids[:2, :11])

for seq in res:
    print(g.decode(seq.tolist()))
