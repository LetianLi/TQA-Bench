import random

import sys

sys.path.append('.')
from symbolic import dataDict
from symDataloader.utils import TaskCore
import symDataloader.testConfig as testConfig
from benchmarkUtils.LLM import gptCall
from benchmarkLoader import singlePrompt

def qaPrompt(dbStr, question, choices):
    totalQuestion = f'{dbStr}\n\n{question}\n\n{choices}'
    prompt = singlePrompt.format(question=totalQuestion)
    return prompt

def gpt5miniCall(dbStr, question, choices):
    prompt = qaPrompt(dbStr, question, choices)
    message, input_tokens, output_tokens = gptCall(
        'gpt-5-mini',
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
    resultPath = f'symDataset/results/TableQA/5_mini{testConfig.saveFileSuffix}.sqlite' # result sqlite
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
            tc.testAll('gpt-5-mini', # The model name saved in taskPath
                    k, # dataset
                    scale, # 8k, 16k, 32k, 64k, 128k
                    False, # if use markdown
                    5, # dbLimit, 10 is ok
                    1, # sampleLimit, 1 is ok
                    14, # questionLimit, 14 is ok
                    gpt5miniCall,
                    timeSleep,
                    injectContextJunk=testConfig.injectContextJunk)
            # tc.testAll('gpt-5-mini', # The model name saved in taskPath
            #         k, # dataset
            #         scale, # 8k, 16k, 32k, 64k, 128k
            #         True, # if use markdown
            #         5, # dbLimit, 10 is ok
            #         1, # sampleLimit, 1 is ok
            #         14, # questionLimit, 14 is ok
            #         gpt5miniCall,
            #         timeSleep)
