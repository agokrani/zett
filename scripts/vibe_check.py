from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

#model_dir = "outputs/qwen2.5-0.5B-instruct-r1-v2-lima-embed-full"

model_dir = "outputs/qwen2.5-0.5B-instruct-r1-v3"
#model_dir = "outputs/phi3.5-mini-instruct-r1"
generation_template = '<｜Assistant｜><think>\\n'

model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer_v3 = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
streamer = TextStreamer(tokenizer)

prompts = [
    [
        # {
        #     "role": "user",
        #     "content": "You are a friendly chatbot.",
        # },
        {
            "role": "user", 
            "content": "A sequence of numbers: 1, 2"
        }
    ], 
    [
        # {
        # "role": "user",
        # "content": "You are a friendly chatbot.",
        # },
        {
            "role": "user", 
            "content": "A, B, C, D, E"
        }
    ], 
    [
        # {
        # "role": "user",
        # "content": "You are a friendly chatbot.",
        # },
        {
            "role": "user", 
            "content": "How many helicopters can a human eat in one sitting? Reply as a thug."
        },
    ],
    [
        {
            "role": "user",
            "content": "Hello!",
        },
    ]
]

print("\nwith chat template applied")

for prompt in prompts:
    model_inputs = tokenizer_v3.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt")
    input_length = model_inputs.shape[1]
    generated_ids = model.generate(model_inputs, eos_token_id=tokenizer.eos_token_id, max_new_tokens=200)
    print(f"Prompt: {prompt}")
    print("\n")
    print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
    print("\n\n")