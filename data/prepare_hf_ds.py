from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

def load_and_process_dataset():
    # Load the LIMO dataset
    dataset = load_dataset("GAIR/LIMO")
    
    # Load tokenizers
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
    
    def process_example(example):
        # Convert to OpenAI messages format
        messages = [{"role": "user", "content": example["question"]}]
        
        # Apply chat templates
        qwen_formatted = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        deepseek_formatted = deepseek_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return {
            "original": example["question"],
            "qwen": qwen_formatted,
            "deepseek": deepseek_formatted
        }
    
    # Process only the train dataset
    processed_dataset = dataset['train'].map(process_example)
    
    # Create lists for each format
    all_texts = (
        [(text, 'original') for text in processed_dataset['original']] +
        [(text, 'qwen') for text in processed_dataset['qwen']] +
        [(text, 'deepseek') for text in processed_dataset['deepseek']]
    )
    
    # Create DataFrame
    df = pd.DataFrame(all_texts, columns=['text', 'format'])
    
    # Save as parquet
    output_file = 'data/train/limo.parquet'
    df.to_parquet(output_file, index=False)
    print(f"Saved combined dataset to {output_file}")
    print(f"Total number of examples: {len(df)}")

if __name__ == "__main__":
    load_and_process_dataset()