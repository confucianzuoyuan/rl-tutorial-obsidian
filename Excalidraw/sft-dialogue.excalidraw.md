---

excalidraw-plugin: parsed
tags: [excalidraw]

---
==⚠  Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. ⚠== You can decompress Drawing data with the command palette: 'Decompress current Excalidraw file'. For more info check in plugin settings under 'Saving'


# Excalidraw Data

## Text Elements
对话格式示意：
<|im_start|>system\n{system_msg}<|im_end|>\n
<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>\n
                                    ↑                           ↑                    ↑
                               user_end                  answer_start           answer_end
                                                           (+3)                   (-1)

掩码设置：
- user_end + 3: 跳过 "<|im_start|>assistant\n" (3个token)
- assistant_end - 1: 不包含 "<|im_end|>" 本身 ^2qvpW9IB

%%
## Drawing
```compressed-json
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
```
%%