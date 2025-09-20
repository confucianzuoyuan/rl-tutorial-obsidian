from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen2.5-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

model.generation_config.do_sample = True
model.generation_config.eos_token_id = [151645, 151643]
model.generation_config.pad_token_id = 151643
model.generation_config.temperature = 0.7
model.generation_config.top_p = 0.8
model.generation_config.top_k = 20
model.generation_config.repetition_penalty = 1.05

tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-Instruct")

history = []
# history.append({"role": "system", "content": "You are a helpful assistant"})
while True:
    question = input('User：' + '\n')
    print('\n')
    history.append({"role": "user", "content": question})

    input_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )
    model_inputs = tokenizer([input_text], return_tensors="pt").to(device)

    if model_inputs.input_ids.size()[1] > 32000:
        break

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1000
    )

    if len(generated_ids) > 32000:
        break

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('Assistant:\n')
    print(response)
    print("--------------------")
    print('\n')
    history.append({"role": "assistant", "content": response})

print("超过模型字数上线，已退出")
