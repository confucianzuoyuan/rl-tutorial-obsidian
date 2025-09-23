[GITHUB地址](https://github.com/yai333/Text-to-SQL-GRPO-Fine-tuning-Pipeline/tree/main)

底座模型：Qwen2.5-Coder-7B-Instruct

任务

- 输入：请帮我写一段SQL，要求查询出部门35岁以上的程序员
- 输出：`SELECT * FROM database WHERE ...`

该模型展现出强大的整体性能，SQL 生成准确率高（44/5 分，得分 4 或 5 分），推理质量优异（48/50 分，得分 4 或 5 分），格式遵循近乎完美（49/50 分，得分 5 分），且具有明确的教育价值。总体而言，88% 的输出得分达到 4.0 分或更高，反映出模型结果的一致性、结构良好且易于解释。

```ad-tip
title: 奖励函数编写要点

1. 使用正则表达式来实现格式奖励
2. 创建一个sqlite3数据库，用来校验SQL语句的正确性
3. 调用DeepSeek或者GPT接口，让外部大语言模型评估一下SQL的质量。
```

