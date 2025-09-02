import torch
from utils import get_question_and_answer, response_generate
from model import device, eos_id, g


question, answer = get_question_and_answer()

print(question.shape, answer.shape)

ppo_model = torch.load('ppo.pt', weights_only=False)
ppo_model.to(device)
ppo_model.eval()

predict = response_generate(ppo_model.actor, question)
predict = predict[:, question.shape[1]:]


correct = 0
for q, a, p in zip(question, answer, predict):
    q, a, p = q.tolist(), a.tolist(), p.tolist()

    if g.encoder['E'] in a:
        split = a.index(eos_id) + 1
        a = a[:split]

    if g.encoder['E'] in p:
        split = p.index(eos_id) + 1
        p = p[:split]

    q, a, p = g.decode(q), g.decode(a), g.decode(p)

    print(q, a, p)

    correct += a == p

print(correct / len(answer))
