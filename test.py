import torch
from llama_cpp import Llama
from llama_cpp.llama import StoppingCriteriaList
from typing import Callable
import numpy as np
import numpy.typing as npt
import time

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Load the model directly from Hugging Face
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
    additional_files=["qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf"],
    n_gpu_layers=-1,  # Use all available GPU layers
    n_ctx=131072,  # Max context length 131072 tokens
    n_batch=512,  # Evaluation Batch Size 512
    n_threads=3,  # CPU Thread Pool Size 3
    use_mmap=True,  # Enable memory mapping
    use_mlock=True,  # Keep model in memory
    offload_kqv=True,  # Offload KV cache to GPU memory
    verbose=True
)

prompt = "Give me a short introduction to large language model."

print("Generating response with llama.cpp:")
print(prompt, end="", flush=True)

# Tokenize the prompt
tokens = llm.tokenize(prompt.encode())

# Create stopping criteria function
max_tokens = 100
def max_tokens_stopping_criteria(input_ids: npt.NDArray[np.intc], logits: npt.NDArray[np.single]) -> bool:
    return input_ids.shape[0] >= max_tokens

# Create stopping criteria list
stopping_criteria = StoppingCriteriaList([max_tokens_stopping_criteria])

# Start timing and token counting
start_time = time.time()
token_count = 0

# Generate tokens with streaming
for token in llm.generate(tokens, top_k=40, top_p=0.95, temp=0.7, repeat_penalty=1.1, stopping_criteria=stopping_criteria):
    # Decode and print the token
    token_text = llm.detokenize([token]).decode('utf-8', errors='ignore')
    print(token_text, end='', flush=True)
    token_count += 1

# Calculate stats
end_time = time.time()
elapsed_time = end_time - start_time
tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0

print(f"\n\n--- Generation Stats ---")
print(f"Total tokens generated: {token_count}")
print(f"Time elapsed: {elapsed_time:.2f} seconds")
print(f"Tokens per second: {tokens_per_second:.2f}")
print("Generation complete!")



