import random
import os
import json

import sys

sys.path.append('.')
from symbolic import dataDict
from symDataloader.utils import TaskCore
import symDataloader.testConfig as testConfig
from benchmarkUtils.LLM import gptCall
from benchmarkLoader import singlePrompt

# ---------------------------------------------------------------------------
# Emulation flag and helpers
# ---------------------------------------------------------------------------
USER_EMULATE: bool = False


def _consume_user_emulate_flag() -> None:
    global USER_EMULATE
    # Support a simple CLI flag without restructuring existing __main__ blocks
    for arg in list(sys.argv[1:]):
        if arg == "--user-emulate":
            USER_EMULATE = True
            try:
                sys.argv.remove(arg)
            except ValueError:
                pass
        else:
            raise Exception(f"Unknown argument: {arg}")


_consume_user_emulate_flag()

def qaPrompt(dbStr, question, choices):
    totalQuestion = f'{dbStr}\n\n{question}\n\n{choices}'
    prompt = singlePrompt.format(question=totalQuestion)
    return prompt

def gpt4oCall(dbStr, question, choices):
    prompt = qaPrompt(dbStr, question, choices)
    
    # Emulation mode: do not call API. Clear console, print request, and capture input as assistant response
    if USER_EMULATE:
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
        except Exception:
            pass
        print("[user-emulate] GPT-4o-mini request:")
        print(f"Model: gpt-4o-mini")
        print(f"Prompt: {prompt}")
        print("\nEnter assistant response:")
        user_resp = input("> ")
        
        # Return the user response with dummy token counts
        return user_resp, 0, 0
    
    message, input_tokens, output_tokens = gptCall(
        'gpt-4o-mini',
        prompt,
        'tmp',
        'symDataset/results/TableQA/log',
        proxies={},
        return_token_counts=True
    )
    return message, input_tokens, output_tokens

if __name__ == '__main__':
    dbRoot = 'symDataset/scaledDB' # path to extract symDataset.zip
    taskPath = 'symDataset/tasks/TableQA/dataset.sqlite' # TableQA's dataset.sqlite
    resultPath = f'symDataset/results/TableQA/4o_mini{testConfig.saveFileSuffix}.sqlite' # result sqlite
    tc = TaskCore(dbRoot, taskPath, resultPath)
    for k in dataDict.keys():
        # Apply table filter if specified
        if testConfig.tableFilter and k not in testConfig.tableFilter:
            continue
        
        for scale in testConfig.dbScales:
            timeSleep = 0
            if scale == '16k':
                # timeSleep = 30
                timeSleep = 0
            elif scale == '32k':
                # timeSleep = 60
                timeSleep = 0
            tc.testAll('gpt-4o-mini', # The model name saved in taskPath
                    k, # dataset
                    scale, # 8k, 16k, 32k, 64k, 128k
                    False, # if use markdown
                    5, # dbLimit, 10 is ok
                    1, # sampleLimit, 1 is ok
                    14, # questionLimit, 14 is ok
                    gpt4oCall,
                    timeSleep,
                    injectContextJunk=testConfig.injectContextJunk)
            # tc.testAll('gpt-4o-mini', # The model name saved in taskPath
            #         k, # dataset
            #         scale, # 8k, 16k, 32k, 64k, 128k
            #         True, # if use markdown
            #         5, # dbLimit, 10 is ok
            #         1, # sampleLimit, 1 is ok
            #         14, # questionLimit, 14 is ok
            #         gpt4oCall,
            #         timeSleep)
