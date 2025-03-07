from typing import List, Optional, Callable
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
# from deepeval.benchmarks.mmlu.task import MMLUTask
# from deepeval.benchmarks import MathQA
# from deepeval.benchmarks import MMLU
from deepeval.benchmarks import GSM8K

def default_answer_parser(text: str) -> Optional[float]:
    """Parse answer from various formats of model outputs."""
    # Handle numbers with commas, scientific notation, and negative signs
    number_pattern = r'-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:e[-+]?\d+)?'
    
    patterns = [
        # XML-style answer tags
        rf'<answer>\s*[$£€]?\s*({number_pattern})\s*</answer>',
        # Complex markdown/code block formats
        rf'(?:```(?:plaintext)?\s*(?:Answer:?)?\s*|(?:\*\*)?(?:The\s+)?(?:[Aa]nswer\s+(?:is|:))?\s*(?:\*\*)?:?\s*)[$£€]?\s*({number_pattern})(?:\s*(?:miles|%))?\s*(?:\n|```|$)',
        # LaTeX formats
        rf'\\\[\s*\\boxed\{{[$£€]?\s*({number_pattern})\}}\s*\\\]',
        # Bold or nested bold formats
        rf'(?:\*\*\*|\*\*)[$£€]?\s*({number_pattern})\s*(?:\*\*\*|\*\*)',
        # Plain formats with possible step numbers
        rf'(?:^|\n)(?:\d+\.\s*)?(?:The\s+)?(?:[Aa]nswer\s+(?:is|:))?\s*[$£€]?\s*({number_pattern})',
        # Simple number on its own line
        rf'(?:^|\n)\s*[$£€]?\s*({number_pattern})\s*$',
        # "is the answer" format
        rf'(?:\d+\.\s*)?({number_pattern})\s*(?:is\s+the\s+(?:final\s+)?answer|$)'
    ]
    
    text = text.strip()
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            number = match.group(1).strip()
            try:
                # Remove commas from numbers like 1,000
                number = number.replace(',', '')
                value = float(number)
                return int(value) if value.is_integer() else value
            except (ValueError, AttributeError):
                continue
    return None

class QwenR1(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        parse_func: Optional[Callable[[str], Optional[float]]] = None,
        eos_token_id=None,
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model = None
        self.tokenizer = None
        self.parse_func = parse_func or default_answer_parser
        self.eos_token_id = eos_token_id

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.eos_token_id is not None: 
            tokenizer.eos_token_id = self.eos_token_id
        return model, tokenizer

    def generate(self, prompt: str) -> str:
        if not self.model:
            self.model, self.tokenizer = self.load_model()

        device = "mps" # the device to load the model onto
        
        # Handle both string prompts and structured prompts
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        prompt = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        self.model.to(device)
        input_length = model_inputs["input_ids"].shape[1]
        generated_ids = self.model.generate(**model_inputs, do_sample=True, eos_token_id=self.tokenizer.eos_token_id, max_new_tokens=2048)
        generated_ids = generated_ids[:, input_length:]
        
        decoded_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if self.parse_func:
            parsed_result = self.parse_func(decoded_output)
            if parsed_result is not None:
                return str(parsed_result)
        
        return decoded_output

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        if not self.model:
            self.model, self.tokenizer = self.load_model()
        device = "mps"  # the device to load the model onto

        # Convert string prompts to chat format and apply template
        formatted_prompts = []
        for prompt in prompts:
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            formatted_prompts.append(formatted_prompt)

        # Tokenize all prompts
        model_inputs = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(device)
        self.model.to(device)

        # Generate for all inputs
        input_length = model_inputs["input_ids"].shape[1]
        generated_ids = self.model.generate(**model_inputs, do_sample=True, eos_token_id=self.tokenizer.eos_token_id, max_new_tokens=500)
        generated_ids = generated_ids[:, input_length:]
        
        decoded_outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        if self.parse_func:
            return [str(self.parse_func(output) or output) for output in decoded_outputs]
        
        return decoded_outputs

    def get_model_name(self):
        return "QwenR1"
    
#qwen_r1 = QwenR1(model_name="outputs/qwen2.5-0.5B-instruct-r1-v2-lima-embed-full", tokenizer_name="deepseek-ai/DeepSeek-V3")
#qwen_r1 = QwenR1(model_name="Qwen/Qwen2.5-0.5B-Instruct", tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct")

#phi4_mini = QwenR1(model_name="microsoft/Phi-4-mini-instruct", tokenizer_name="microsoft/Phi-4-mini-instruct", eos_token_id=200020)
phi4_mini_r1 = QwenR1("outputs/phi4-mini-instruct-r1", "outputs/phi4-mini-instruct-r1", eos_token_id=1)
#benchmark = MMLU(tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY])

#esults = benchmark.evaluate(model=qwen_r1)

#benchmark = MathQA()

# Modify the benchmark initialization to avoid schema validation
benchmark = GSM8K(n_shots=8, confinement_instructions="Make sure to output only the numerical answer in the following fomat: \n\n**Answer**: <number here>\n\n Please follow the instructions carefully and output format properly to get the correct score.")

benchmark.evaluate(model=phi4_mini_r1)

print(benchmark.overall_score)
print(benchmark.predictions)

import pdb;pdb.set_trace()
