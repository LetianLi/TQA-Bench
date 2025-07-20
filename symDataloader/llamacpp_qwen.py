"""
Wrapper for Qwen2.5-7B-Instruct using llama.cpp directly.

Use `uv sync` to install the dependencies.
"""

import random
import sys
import time
from llama_cpp import Llama
from llama_cpp.llama import StoppingCriteriaList
from typing import Callable, List
from llama_cpp.llama_types import ChatCompletionRequestMessage
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

sys.path.append('.')
from symbolic import dataDict
from symDataloader.utils import TaskCore
from benchmarkLoader import singlePrompt

# ---------------------------------------------------------------------------
# 1. One global model handle (re-used for every call)
# ---------------------------------------------------------------------------
_MODEL = None

def get_model():
    """Initialize and return the llama.cpp model instance."""
    global _MODEL
    if _MODEL is None:
        _MODEL = Llama.from_pretrained(
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
            verbose=False
        )
    return _MODEL

# ---------------------------------------------------------------------------
# 2. Prompt helper (same as before)
# ---------------------------------------------------------------------------
def qaPrompt(dbStr, question, choices):
    totalQuestion = f'{dbStr}\n\n{question}\n\n{choices}'
    prompt = singlePrompt.format(question=totalQuestion)
    return prompt

# ---------------------------------------------------------------------------
# 3. Public API – plug into TaskCore
# ---------------------------------------------------------------------------
def qwenLlamaCppCall(dbStr, question, choices):
    """
    Runs one inference via llama.cpp and returns the raw completion string.
    """
    prompt = qaPrompt(dbStr, question, choices)
    max_tokens = 10000

    # Get model instance
    llm = get_model()

    # Create progress bars
    token_pbar = tqdm(total=max_tokens, desc="      Tokens", position=4, leave=False, unit="tokens")

    # Create messages for chat completion
    messages: List[ChatCompletionRequestMessage] = [
        {"role": "user", "content": prompt}
    ]

    # Start timing and token counting
    start_time = time.time()
    token_count = 0
    full_response = ""

    # Use create_chat_completion with streaming
    response_stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repeat_penalty=1.1,
        stream=True,
        stop=["I AM DONE"]
    )

    # Process the streaming response
    for chunk in response_stream:
        if chunk["choices"][0]["delta"].get("content"): # type: ignore
            content: str = chunk["choices"][0]["delta"]["content"] # type: ignore
            full_response += content
            token_count += 1
            token_pbar.n = token_count
            token_pbar.refresh()

    # Close progress bar
    token_pbar.close()

    # For streaming responses, we need to estimate token counts
    # Since we can't get usage from stream chunks, we'll use our manual count
    input_tokens = len(llm.tokenize(prompt.encode()))
    output_tokens = token_count

    return full_response.strip(), input_tokens, output_tokens

if __name__ == '__main__':
    # Cache the model
    get_model()
    print("Model cached\n\n")
    
    # Interactive mode
    if len(sys.argv) > 1 and (sys.argv[1] == "--interactive" or sys.argv[1] == "interactive"):
        print("Qwen2.5-7B-Instruct Chat (llama.cpp)")
        print("Loading model...")
        
        # Initialize model
        llm = get_model()
        
        while True:
            user_input = input("\nUser: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            
            # Create messages for chat completion
            messages: List[ChatCompletionRequestMessage] = [
                {"role": "user", "content": user_input}
            ]
            
            print("Assistant: ", end="", flush=True)
            
            # Stream the response
            full_response = ""
            token_count = 0
            
            try:
                # Use create_chat_completion with streaming
                response_stream = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=20,
                    repeat_penalty=1.1,
                    stream=True,
                    stop=["I AM DONE"]
                )
                
                # Process the streaming response
                for chunk in response_stream:
                    if chunk["choices"][0]["delta"].get("content"): # type: ignore
                        content: str = chunk["choices"][0]["delta"]["content"] # type: ignore
                        full_response += content
                        token_count += 1
                        print(content, end="", flush=True)
                
                print(f"\n[Response length: {len(full_response)} chars, {token_count} output tokens]")
                
            except Exception as e:
                print(f"\nError: {e}")
                print("Make sure the model files are available.")
                raise e
    else:
        # Test structure similar to original
        dbRoot = 'symDataset/scaledDB' # path to extract symDataset.zip
        taskPath = 'symDataset/tasks/TableQA/dataset.sqlite' # TableQA's dataset.sqlite
        resultPath = 'symDataset/results/TableQA/llamacpp_qwen2.5.sqlite' # result sqlite
        tc = TaskCore(dbRoot, taskPath, resultPath)
        for k in dataDict.keys():
            # for scale in ['8k', '16k', '32k', '64k']:
            for scale in ['8k']:
                timeSleep = 0
                # if scale == '16k':
                #     timeSleep = 30
                # elif scale == '32k':
                #     timeSleep = 60
                tc.testAll('Qwen2.5-7B-Instruct-Local', # The model name saved in taskPath
                        k, # dataset
                        scale, # 8k, 16k, 32k, 64k, 128k
                        False, # if use markdown
                        5, # dbLimit, 10 is ok
                        1, # sampleLimit, 1 is ok
                        14, # questionLimit, 14 is ok
                        qwenLlamaCppCall,
                        timeSleep)
                tc.testAll('Qwen2.5-7B-Instruct-Local', # The model name saved in taskPath
                        k, # dataset
                        scale, # 8k, 16k, 32k, 64k, 128k
                        True, # if use markdown
                        5, # dbLimit, 10 is ok
                        1, # sampleLimit, 1 is ok
                        14, # questionLimit, 14 is ok
                        qwenLlamaCppCall,
                        timeSleep) 