import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from tqdm import tqdm
import threading
import time

import sys
sys.path.append('.')
from symbolic import dataDict
from symDataloader.utils import TaskCore
from benchmarkLoader import singlePrompt

class QwenTransformersChat:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device=None, max_context=8192):
        print(f"Initializing QwenTransformersChat...")
        print(f"  Model: {model_name}")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Fix pad token issue - set pad_token to eos_token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"  Set pad_token to eos_token: {self.tokenizer.eos_token}")
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        self.max_context = max_context
        self.messages = []  # List of {"role": str, "content": str}
        
        print(f"  Device: {self.device}")
        print(f"  Max context: {self.max_context}")
        print(f"  Vocab size: {self.tokenizer.vocab_size}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("QwenTransformersChat initialized successfully!")

    def reset(self):
        self.messages = []

    def append_turn(self, role, content):
        self.messages.append({"role": role, "content": content})

    def _build_inputs(self, add_generation_prompt=True):
        # Use chat template for prompt formatting
        input_ids = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt"
        ).to(self.device)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        return input_ids, attention_mask

    def _count_tokens(self, input_ids):
        return input_ids.shape[-1]

    def _truncate_history(self):
        # Remove oldest turns until context fits
        while True:
            input_ids, _ = self._build_inputs()
            n_tokens = self._count_tokens(input_ids)
            if n_tokens <= self.max_context:
                break
            if len(self.messages) > 1:
                self.messages.pop(0)
            else:
                break

    def chat(self, user_input, max_new_tokens=256, temperature=0.7, top_p=0.9, show_tqdm=True):
        self.append_turn("user", user_input)
        self._truncate_history()
        input_ids, attention_mask = self._build_inputs()
        n_tokens = self._count_tokens(input_ids)
        
        if show_tqdm:
            print(f"Context usage: {n_tokens}/{self.max_context} tokens ({n_tokens/self.max_context*100:.1f}%)")
        
        # Create streamer for real-time output
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        # Start generation in a separate thread
        generation_thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        generation_thread.start()
        
        # Display streaming output with progress
        response = ""
        token_count = 0
        
        if show_tqdm:
            progress_bar = tqdm(total=max_new_tokens, desc="Generating", unit="tokens", leave=False)
        
        try:
            for new_text in streamer:
                response += new_text
                token_count += 1
                
                if show_tqdm:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"tokens": token_count})
        
        except Exception as e:
            print(f"Error during generation: {e}")
            response = f"Error: {e}"
        
        finally:
            if show_tqdm:
                progress_bar.close()
        
        # Wait for generation to complete
        generation_thread.join()
        
        self.append_turn("assistant", response.strip())
        return response.strip()

    def single_turn(self, user_input, **kwargs):
        self.reset()
        return self.chat(user_input, **kwargs)

    def count_tokens(self, text):
        """Count tokens in a text string"""
        return len(self.tokenizer.encode(text))

# ---------------------------------------------------------------------------
# Global model instance (reused for every call)
# ---------------------------------------------------------------------------
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = QwenTransformersChat()
    return _MODEL

# ---------------------------------------------------------------------------
# Prompt helper (same as lmstudio_qwen.py)
# ---------------------------------------------------------------------------
def qaPrompt(dbStr, question, choices):
    totalQuestion = f'{dbStr}\n\n{question}\n\n{choices}'
    prompt = singlePrompt.format(question=totalQuestion)
    return prompt

# ---------------------------------------------------------------------------
# Public API – plug into TaskCore
# ---------------------------------------------------------------------------
def qwenLocalCall(dbStr, question, choices):
    """
    Runs one inference via Transformers and returns the raw completion string with token counts.
    """
    model = get_model()
    prompt = qaPrompt(dbStr, question, choices)
    
    # Count input tokens
    input_tokens = model.count_tokens(prompt)
    
    # Create progress bars
    prompt_pbar = tqdm(total=100, desc="      Processing", position=3, leave=False)
    prompt_pbar.update(100)  # Immediate completion for transformers
    
    # Generate response
    response = model.single_turn(prompt, max_new_tokens=512, temperature=0.6, top_p=0.9, show_tqdm=False)
    
    # Count output tokens
    output_tokens = model.count_tokens(response)
    
    # Close progress bars
    prompt_pbar.close()
    
    return response.strip(), input_tokens, output_tokens

if __name__ == "__main__":
    # Example usage for interactive testing
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        print("Qwen2.5-7B-Instruct Chat (Transformers)")
        chat = QwenTransformersChat()
        while True:
            user_input = input("\nUser: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            response = chat.chat(user_input)
            print(f"Assistant: {response}")
            print(f"[Response length: {len(response)} chars, {chat.count_tokens(response)} tokens]")
    else:
        # Test structure similar to lmstudio_qwen.py
        dbRoot = 'symDataset/scaledDB' # path to extract symDataset.zip
        taskPath = 'symDataset/tasks/TableQA/dataset.sqlite' # TableQA's dataset.sqlite
        resultPath = 'symDataset/results/TableQA/transformers_qwen2.5.sqlite' # result sqlite
        tc = TaskCore(dbRoot, taskPath, resultPath)
        for k in dataDict.keys():
            # for scale in ['8k', '16k', '32k', '64k']:
            for scale in ['8k']:
                timeSleep = 0
                # if scale == '16k':
                #     timeSleep = 30
                # elif scale == '32k':
                #     timeSleep = 60
                tc.testAll('Qwen2.5-7B-Instruct-Transformers', # The model name saved in taskPath
                        k, # dataset
                        scale, # 8k, 16k, 32k, 64k, 128k
                        False, # if use markdown
                        5, # dbLimit, 10 is ok
                        1, # sampleLimit, 1 is ok
                        14, # questionLimit, 14 is ok
                        qwenLocalCall,
                        timeSleep)
                tc.testAll('Qwen2.5-7B-Instruct-Transformers', # The model name saved in taskPath
                        k, # dataset
                        scale, # 8k, 16k, 32k, 64k, 128k
                        True, # if use markdown
                        5, # dbLimit, 10 is ok
                        1, # sampleLimit, 1 is ok
                        14, # questionLimit, 14 is ok
                        qwenLocalCall,
                        timeSleep) 