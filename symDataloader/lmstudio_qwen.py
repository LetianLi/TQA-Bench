"""
Wrapper for Qwen2.5-7B-Instruct served by LM Studio's local server.

Use `uv sync` to install the dependencies.
"""

import random
import sys
import lmstudio as lms
import re
from tqdm import tqdm

sys.path.append('.')
from symbolic import dataDict
from symDataloader.utils import TaskCore
from benchmarkLoader import singlePrompt

# ---------------------------------------------------------------------------
# 1. One global model handle (re-used for every call)
# ---------------------------------------------------------------------------
lms.configure_default_client("localhost:5841")
# _MODEL = lms.llm("qwen/qwen3-8b")   # alias used by LM Studio catalog
_MODEL = lms.llm("qwen/qwen2.5-7b-instruct")   # alias used by LM Studio catalog

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
def qwenLocalCall(dbStr, question, choices):
    """
    Runs one inference via LM Studio and returns the raw completion string.
    """
    prompt = qaPrompt(dbStr, question, choices)
    maxTokens = 30000

    # Create progress bars
    prompt_pbar = tqdm(total=100, desc="      Prompt", position=3, leave=False)
    token_pbar = tqdm(total=maxTokens, desc="      Tokens", position=4, leave=False, unit="tokens")

    # Track prompt processing progress
    def on_prompt_progress(progress):
        prompt_pbar.n = int(progress * 100)
        prompt_pbar.refresh()

    # Stream the completion
    prediction_stream = _MODEL.complete_stream(
        prompt,
        config={ # Best parameters according to https://huggingface.co/Qwen/Qwen3-8B#best-practices
            "maxTokens": maxTokens,
            "temperature": 0.6,
            "topPSampling": 0.95,
            "topKSampling": 20,
            "minPSampling": 0,
            "repeatPenalty": 1.1,
            "stopStrings": ["I AM DONE"]
        },
        on_prompt_processing_progress=on_prompt_progress
    )

    # Process the stream
    full_response = ""
    token_count = 0
    
    for fragment in prediction_stream:
        if hasattr(fragment, "content"):
            full_response += fragment.content
            token_count += 1
            token_pbar.n = token_count
            token_pbar.refresh()

    # Close progress bars
    prompt_pbar.close()
    token_pbar.close()

    # Get final result for stats
    result = prediction_stream.result()
    
    # Get token information
    input_tokens = result.stats.prompt_tokens_count
    output_tokens = result.stats.predicted_tokens_count

    return full_response.strip(), input_tokens, output_tokens
    

if __name__ == '__main__':
    # Example usage for interactive testing
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        print("Qwen2.5-7B-Instruct Chat (LM Studio)")
        print("Connected to LM Studio server at localhost:5841")
        
        while True:
            user_input = input("\nUser: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            
            # Create a simple prompt for chat
            prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            
            print("Assistant: ", end="", flush=True)
            
            # Stream the response
            full_response = ""
            token_count = 0
            
            try:
                prediction_stream = _MODEL.complete_stream(
                    prompt,
                    config={
                        "maxTokens": 512,
                        "temperature": 0.7,
                        "topPSampling": 0.9,
                        "topKSampling": 20,
                        "minPSampling": 0,
                        "repeatPenalty": 1.1,
                    }
                )
                
                for fragment in prediction_stream:
                    if hasattr(fragment, "content"):
                        full_response += fragment.content
                        token_count += 1
                        print(fragment.content, end="", flush=True)
                
                print(f"\n[Response length: {len(full_response)} chars, ~{token_count} tokens]")
                
            except Exception as e:
                print(f"\nError: {e}")
                print("Make sure LM Studio is running with the Qwen2.5-7B-Instruct model loaded.")
    else:
        # Test structure similar to original
        dbRoot = 'symDataset/scaledDB' # path to extract symDataset.zip
        taskPath = 'symDataset/tasks/TableQA/dataset.sqlite' # TableQA's dataset.sqlite
        resultPath = 'symDataset/results/TableQA/lmstudio_qwen2.5.sqlite' # result sqlite
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
                        qwenLocalCall,
                        timeSleep)
                tc.testAll('Qwen2.5-7B-Instruct-Local', # The model name saved in taskPath
                        k, # dataset
                        scale, # 8k, 16k, 32k, 64k, 128k
                        True, # if use markdown
                        5, # dbLimit, 10 is ok
                        1, # sampleLimit, 1 is ok
                        14, # questionLimit, 14 is ok
                        qwenLocalCall,
                        timeSleep)
