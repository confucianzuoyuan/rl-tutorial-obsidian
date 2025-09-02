import torch
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel, BertConfig, BertModel
from data_generator import DataGenerator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

g = DataGenerator()
bos_id = g.encoder['S']
eos_id = g.encoder['E']
pad_id = g.encoder['P']
vocab_size = len(g.decoder)


class LLM_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = GPT2Config(bos_token_id=bos_id,
                                 eos_token_id=eos_id,
                                 n_embd=64,
                                 n_head=4,
                                 n_layer=4,
                                 n_positions=128,
                                 vocab_size=vocab_size)

        self.feature = GPT2Model(self.config)
        self.fc_out = torch.nn.Linear(64, self.config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask):
        out = self.feature(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state

        return self.fc_out(out)

# 输出回答


class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = BertConfig(hidden_size=64,
                                 intermediate_size=64,
                                 max_position_embeddings=128,
                                 num_attention_heads=4,
                                 num_hidden_layers=4,
                                 vocab_size=vocab_size)

        self.feature = BertModel(self.config)
        self.fc_out = torch.nn.Linear(self.config.hidden_size, 4)

    def forward(self, input_ids, attention_mask):
        out = self.feature(input_ids=input_ids,
                           attention_mask=attention_mask).pooler_output

        return self.fc_out(out)


class PPO_Model(torch.nn.Module):
    def __init__(self, llm_model):
        super().__init__()
        # 演员模型，也就是策略模型是我们的大语言模型
        self.actor = llm_model
        # 评论家模型，也就是价值函数模型，用来评估演员输出的回答的价值
        self.critic = torch.nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask):
        # 大模型输出的回答
        last_hidden_state = self.actor.feature(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True).last_hidden_state

        logits = self.actor.fc_out(last_hidden_state)
        # 对回答进行评分
        value = self.critic(last_hidden_state).squeeze(-1)

        return logits, value  # 回答，评分
