import torch
from model import device, g, LLM_Model, RewardModel
from utils import get_answer, get_question, get_reward

llm_model = LLM_Model()
llm_model = torch.load('llm.pt', weights_only=False)
llm_model.to(device)
llm_model.eval()  # 设置为评估模式


reward_model = RewardModel()
reward_model = torch.load('rm.pt', weights_only=False)
reward_model.to(device)
reward_model.eval()  # 切换到评估模式


label, question = get_question()
answer = get_answer(llm_model, question)
reward = get_reward(reward_model, question, answer, label)

for i in range(10):
    print('问题：', g.decode(question[i]), '; 答案：', g.decode(
        answer[i]), '; 奖励：', reward[i].item())
