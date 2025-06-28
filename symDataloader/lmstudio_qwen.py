"""
Wrapper for Qwen2.5-7B-Instruct served by LM Studio's local server.

Requires:
    pip install lmstudio
    # and LM Studio desktop running, with the model downloaded:
    #   lms get qwen2.5-7b-instruct-mlx
"""

import random
import sys
import lmstudio as lms
import re

sys.path.append('.')
from symbolic import dataDict
from symDataloader.utils import TaskCore
from benchmarkLoader import singlePrompt

# ---------------------------------------------------------------------------
# 1. One global model handle (re-used for every call)
# ---------------------------------------------------------------------------
lms.configure_default_client("localhost:5841")
_MODEL = lms.llm("qwen2.5-7b-instruct-mlx")   # alias used by LM Studio catalog

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

    result = _MODEL.complete(
        prompt,
        config={
            "maxTokens": 1500,
            "temperature": 0.85,
            "topPSampling": 0.9,
        },
    )

    # `result` behaves like a str; avoid SDK version differences:
    text = result.content if hasattr(result, "content") else str(result)
    return text.strip()
    

if __name__ == '__main__':
    dbRoot = 'symDataset/scaledDB' # path to extract symDataset.zip
    taskPath = 'symDataset/tasks/TableQA/dataset.sqlite' # TableQA's dataset.sqlite
    resultPath = 'symDataset/results/TableQA/lmstudio_qwen.sqlite' # result sqlite
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
