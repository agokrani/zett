from transformers import AutoModelForCausalLM, AutoTokenizer
from deepfocus import FOCUS

source_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
source_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

target_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")


target_embeddings = FOCUS(
    source_embeddings=source_model.get_input_embeddings().weight,
    source_tokenizer=source_tokenizer,
    target_tokenizer=target_tokenizer,
    auxiliary_embedding_mode="fasttext-tokenlevel",
    target_training_data_path="data/train-4000-all-limo.txt",
    ood_init_method="average", 
    special_tokens_source_to_target_mapping={
        "<|im_start|>": "<｜begin▁of▁sentence｜>",
        "<|im_end|>": "<｜end▁of▁sentence｜>", 
        "<|im_start|>user\n": "<｜User｜>",
        "<|im_start|>assistant\n": "<｜Assistant｜>",
    }
)

source_model.resize_token_embeddings(len(target_tokenizer))
source_model.get_input_embeddings().weight.data = target_embeddings

source_model.save_pretrained("outputs/qwen2.5-0.5B-instruct-r1-v3")
target_tokenizer.save_pretrained("outputs/qwen2.5-0.5B-instruct-r1-v3")
