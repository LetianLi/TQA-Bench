import sys
import os

sys.path.append('.')
from symbolic import dataDict
from symDataloader.utils import TaskCore


# Global counter to track how many requests were "fired"
REQUEST_COUNTER = 0


def dryrunCall(dbStr, question, choices):
    """Return an incrementing request number and zero token counts.

    The return shape matches the expected interface: (message, input_tokens, output_tokens).
    """
    global REQUEST_COUNTER
    REQUEST_COUNTER += 1
    return str(REQUEST_COUNTER), 0, 0


if __name__ == '__main__':
    dbRoot = 'symDataset/scaledDB'  # path to extract symDataset.zip
    taskPath = 'symDataset/tasks/TableQA/dataset.sqlite'  # TableQA's dataset.sqlite
    resultPath = 'symDataset/results/TableQA/dryrun.sqlite'  # result sqlite
    if '--restart' in sys.argv[1:]:
        try:
            os.remove(resultPath)
        except FileNotFoundError:
            pass
    tc = TaskCore(dbRoot, taskPath, resultPath)
    try:
        for k in dataDict.keys():
            # for scale in ['8k', '16k', '32k', '64k']:
            for scale in ['8k']:
                timeSleep = 0
                if scale == '16k':
                    timeSleep = 0
                elif scale == '32k':
                    timeSleep = 0
                tc.testAll(
                    'dryrun',  # The model name saved in taskPath/results
                    k,  # dataset
                    scale,  # 8k, 16k, 32k, 64k, 128k
                    False,  # if use markdown
                    5,  # dbLimit, 10 is ok
                    1,  # sampleLimit, 1 is ok
                    14,  # questionLimit, 14 is ok
                    dryrunCall,
                    timeSleep,
                )
                tc.testAll(
                    'dryrun',  # The model name saved in taskPath/results
                    k,  # dataset
                    scale,  # 8k, 16k, 32k, 64k, 128k
                    True,  # if use markdown
                    5,  # dbLimit, 10 is ok
                    1,  # sampleLimit, 1 is ok
                    14,  # questionLimit, 14 is ok
                    dryrunCall,
                    timeSleep,
                )
    except KeyboardInterrupt:
        print(f"\nInterrupted. Total requests succeeded: {REQUEST_COUNTER}")
        sys.exit(130)
    print(f"Total requests succeeded: {REQUEST_COUNTER}")
