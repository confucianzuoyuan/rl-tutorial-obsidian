import torch
from model import g, PPO_Model, RewardModel, device, pad_id, eos_id, bos_id
from utils import logprobs_from_logits, masked_whiten, masked_mean, get_answer, get_question, get_reward


# 需要微调的模型
ppo_model = PPO_Model(torch.load('llm.pt', weights_only=False))
ppo_model.to(device)
ppo_model.train()
# 冻结的参考模型
ppo_ref_model = PPO_Model(torch.load('llm.pt', weights_only=False))
ppo_ref_model.to(device)
# 奖励模型
reward_model = RewardModel()
reward_model = torch.load('rm.pt', weights_only=False)
reward_model.to(device)

# 冻结参考模型的参数
for i in ppo_ref_model.parameters():
    i.requires_grad_(False)
ppo_ref_model.eval()
# 冻结奖励模型的参数
for i in reward_model.parameters():
    i.requires_grad_(False)
reward_model.eval()


class PPOTrainer:
    def __init__(self):
        self.optimizer = torch.optim.Adam(ppo_model.parameters(), lr=1e-5)

    def step(self, question, answer, reward):
        with torch.no_grad():
            # 编码数据
            data = [q.tolist() + a.tolist() for q, a in zip(question, answer)]
            input_ids, attention_mask = g.batch_pad(data)
            del data
            input_ids = torch.LongTensor(input_ids).to(device)
            attention_mask = torch.LongTensor(attention_mask).to(device)

            # question和answer只需要长度
            question_length = [question.shape[1]] * len(question)
            answer_length = []

            for a in answer:
                if eos_id in a:
                    # `E` 后面的 padding 标记不要
                    answer_length.append(a.tolist().index(eos_id) + 1)
                    continue
                answer_length.append(len(a))

            del question
            del answer

            # 根据 question 计算生成 answer 的概率
            # 并计算每个动作（也就是输出）的分数（也就是价值）
            prob_log, value, mask = self.batched_forward_pass(
                ppo_model,
                input_ids,
                attention_mask,
                question_length,
                answer_length
            )

            # 使用冻结的ref参考模型，计算冻结的模型输出的概率，
            # 为了计算 KL 散度
            prob_log_ref, _, _ = self.batched_forward_pass(
                ppo_ref_model,
                input_ids,
                attention_mask,
                question_length,
                answer_length
            )

            # 计算正在微调的模型的输出和冻结的参考模型的输出之间的 KL 散度
            # 然后融入到 reward 奖励中。
            reward = self.compute_rewards(
                reward,
                prob_log,
                prob_log_ref,
                mask
            )

            # 计算价值，优势和回报（收益）
            values, advantages, returns = self.compute_advantages(
                value,
                reward,
                mask
            )

        # 每批数据循环 N 次模型
        # 假设一批数据有 64 条问答数据，循环 4 轮
        # 那么 ppo_model 会被更新 4 x 64 = 256 次。
        for _ in range(4):
            # 每次计算一条问答数据
            for i in range(len(input_ids)):
                # 计算单条问答的概率和价值
                prob_log_new, value_new, _ = self.batched_forward_pass(
                    ppo_model,
                    input_ids[i].unsqueeze(0),
                    attention_mask[i].unsqueeze(0),
                    [question_length[i]],
                    [answer_length[i]]
                )

                # 根据新旧概率求出变化率，也就是重要性采样
                # 进而求出 loss
                loss = self.get_loss(
                    prob_log[i].unsqueeze(0),
                    values[i].unsqueeze(0),
                    prob_log_new,
                    value_new,
                    mask[i].unsqueeze(0),
                    advantages[i].unsqueeze(0),
                    returns[i].unsqueeze(0)
                )

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def batched_forward_pass(
        self,
        model,
        input_ids,
        attention_mask,
        question_length,
        answer_length
    ):
        # 输出的回答和回答的价值
        logits, value = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 取每个 token 也就是每个字符的对数概率
        prob_log = logprobs_from_logits(logits[:, :-1], input_ids[:, 1:])

        # 将预测结果部分的掩码设置为 1 。
        # PAD 部分的掩码都设置为 0 。
        mask = torch.zeros_like(attention_mask)
        mask[:, :-1] = attention_mask[:, 1:]
        for i in range(len(input_ids)):
            start = question_length[i] - 1
            end = start + answer_length[i]
            mask[i, :start] = 0
            mask[i, end:] = 0

        # 对最后一个字符的预测没有意义，直接丢弃
        value = value[:, :-1]
        mask = mask[:, :-1]

        return prob_log, value, mask

    def compute_rewards(self, reward, prob_log, prob_log_ref, mask):
        reward_kl = []  # 正则项是kl散度

        for i in range(len(reward)):
            # 求 `ppo_model` 和 `ppo_ref_model` 输出的对数概率的 KL 散度
            # 这里的 KL 散度的计算非常简单，就是 `差值 x -0.2`
            kl = (prob_log[i] - prob_log_ref[i]) * -0.2

            # 把 `奖励模型的评分` 加到一条数据的最后一个字符的 kl 散度上面
            # 先找出最后一个字符的索引，也就是掩码不为 0 的索引
            if (mask[i] == 0).all():
                idx = 0
            else:
                idx = mask[i].nonzero()[-1].item()
            kl[idx] += reward[i]

            # 将最终得到的奖励添加到列表中
            reward_kl.append(kl)

        return torch.stack(reward_kl)

    def compute_advantages(
        self,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        mask: torch.FloatTensor,
    ):
        lastgaelam = 0
        advantages_reversed = []
        # 生成的回答的长度
        gen_len = rewards.shape[-1]

        # 将 掩码 为 0 的地方的价值和奖励设置为 0
        values = values * mask
        rewards = rewards * mask

        gamma = 0.95  # 折扣因子的大小
        lam = 1  # 平衡因子

        # 使用广义优势估计中的优势计算的递推公式，逆向计算
        # 也就是先计算 A_{t+2} , 然后保存 A_{t+2} ，
        # 然后再利用 A_{t+2} 计算 A{t+1} ，以此类推
        for t in reversed(range(gen_len)):
            # 计算 V(s_{t+1})
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            # 计算 delta，
            # 使用GAE中的公式：
            # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            # lastgaelam 中保存的是 A_{t+1}
            # 下一行代码运行完 lastgaelam ==> A_t
            # 保存下来，下次迭代就可以计算 A_{t-1} 了
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        # 将优势恢复成顺序 `[A_t, A_{t+1}, A_{t+2}, ...]`
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        # 优势 Advantage = Q - V 所以 Q = Advantage + V
        returns = advantages + values
        # 对 `advantages`（优势函数估计）进行归一化处理，
        # 通常是去均值、除以标准差，使其分布更稳定。
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()
        return values, advantages, returns

    def get_loss(
        self,
        prob_log_old,  # 旧策略输出的对数概率
        values,  # 旧策略动作的价值
        prob_log_new,  # 新策略输出的对数概率
        value_new,  # 新策略动作的价值
        mask,
        advantages,  # 使用旧策略计算的优势
        returns  # 使用旧策略计算的 状态-动作价值函数 Q
    ):
        # 计算比率
        ratio = torch.exp(prob_log_new - prob_log_old)

        # 计算价值函数模型的损失
        # 注意：我们的价值函数模型就在 PPO_Model 中。是一个线性层。用来评估输出的价值。
        # 专业名称叫做：value_head
        # 使用均方误差的损失mse
        vf_losses1 = (value_new - returns) ** 2  # (V(s)-Q(s,a))^2
        # 将新策略输出的价值裁剪到范围 (values - 0.2, values + 0.2)
        vf_losses2 = torch.clip(value_new, values - 0.2, values + 0.2)
        # 计算第二份 mse
        vf_losses2 = (vf_losses2 - returns) ** 2
        # 取最大的一份 loss 然后归一化
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)

        # 计算 ppo 的 loss
        # 在最大化目标函数相当于最小化 `-目标函数` ，所以需要加负号
        ppo_losses1 = -advantages * ratio
        # 公式中对比率的裁剪 `(1-\epsilon, 1+\epsilon)`
        ppo_losses2 = -advantages * torch.clip(ratio, 1.0 - 0.2, 1.0 + 0.2)
        # 取最大的一份 loss 然后归一化
        ppo_loss = masked_mean(torch.max(ppo_losses1, ppo_losses2), mask)

        # 最终的损失 = 演员(策略模型)的损失 + 0.1 * 评论家(价值模型，线性层)的损失
        # 一个反向传播就把actor和critic都更新了。
        # actor 和 critic 共享了基础模型，这样节省算力。共享基础模型一致性也更好，
        # 因为使用了同样的上下文
        return ppo_loss + 0.1 * vf_loss


trainer = PPOTrainer()

for epoch in range(2000):
    label, question = get_question()
    answer = get_answer(ppo_model.actor, question)
    reward = get_reward(reward_model, question, answer, label)

    trainer.step(question, answer, reward)

    if epoch % 100 == 0:
        print(epoch, reward.mean().item())
        for _, q, a, r in zip(range(2), question, answer, reward):
            q = g.decode(q.tolist())
            a = g.decode(a.tolist())
            r = r.item()
            print(q, a, r)

torch.save(ppo_model, 'ppo.pt')
