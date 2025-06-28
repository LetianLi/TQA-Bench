import os
import tiktoken
import requests
from datetime import datetime
from uuid import uuid4

import sys
sys.path.append('.')
from benchmarkUtils.database import DB
from benchmarkUtils.jsTool import JS

def gptCall(model,
            prompt,
            logStart="",
            logPath="logs",
            proxies={
                'http': 'socks5://127.0.0.1:1080',
                'https': 'socks5://127.0.0.1:1080',
            }, # Proxy dictionary; defaults to a SOCKS5 proxy on port 1080
            OPENAI_API_KEY=None,
            otherInfo={},
            delPrompt=True
            ):
    """
    model: GPT model, e.g. gpt-4, gpt-4o, gpt-4o-mini
    prompt: prompt text
    logStart: prefix for log file names; avoid inserting '_' in it
    logPath: directory for log files; created automatically if it doesn't exist
    proxies: proxy configuration for requests; default is a SOCKS5 proxy on port 1080
    OPENAI_API_KEY: OpenAI API key; if None, read from environment variable
    """
    os.makedirs(logPath, exist_ok=True)
    if OPENAI_API_KEY is None:
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }
    bodies = {
        "model": model,
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": prompt
            }
            ]
        }
        ],
    }
    msg = None
    try:
        msg = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=bodies, proxies=proxies).json()
        # msg = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=bodies).json()
        msg = msg['choices'][0]['message']['content']
    except Exception as e:
        raise Exception(str(msg))
    logInfo = {"model": model, "prompt": prompt, "message": msg}
    if delPrompt:
        del logInfo['prompt']
    logInfo.update(otherInfo)
    fileName = datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + "_" + str(uuid4()) + ".json"
    if logStart != "":
        fileName = logStart + "_" + fileName
    filePath = os.path.join(logPath, fileName)
    JS(filePath).newJS(logInfo)
    return msg

def countDBToken(dbPath, markdown=False):
    """
    dbPath: path to the SQLite file
    markdown: whether to convert to Markdown format (otherwise CSV)
    Counts the token size of the tables in the SQLite file after serialization
    """
    if not os.path.isfile(dbPath):
        return 0
    db = DB(dbPath)
    dbStr = db.defaultSerialization(markdown=markdown)
    tkTool = tiktoken.get_encoding("cl100k_base")
    return len(tkTool.encode(dbStr))

def countDFToken(df, markdown=False):
    dfStr = ''
    if markdown:
        dfStr = df.to_markdown(index=False)
    else:
        dfStr = df.to_csv(index=False)
    tkTool = tiktoken.get_encoding("cl100k_base")
    return len(tkTool.encode(dfStr))

if __name__ == '__main__':
    # dbp = 'dataset/sampleDB/movie/movie.sqlite'
    # logPath = 'dataset/log/tmp/'
    # db = DB(dbp)
    # dbStr = db.defaultSerialization(markdown=True)
    # prompt = f'Please summarize the important information in the following tables.\n\n{dbStr}'
    # res = gptCall('gpt-4o-mini', prompt, logPath=logPath)
    # print(res)
    res = gptCall('gpt-4o-mini', 'I love you.', 'adsfa', 'tmp')
    print(res)
