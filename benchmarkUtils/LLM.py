import os
import tiktoken
import requests
import time
import random
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv
import sys
sys.path.append('.')
from benchmarkUtils.database import DB
from benchmarkUtils.jsTool import JS
from tqdm import tqdm

load_dotenv()

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
            delPrompt=True,
            return_token_counts=False
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
    message = None
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens = 0

    def _parse_retry_after_seconds(resp, data):
        # Prefer standard Retry-After header (seconds or HTTP-date)
        ra = resp.headers.get('Retry-After') if resp is not None else None
        if ra:
            try:
                return float(ra)
            except Exception:
                # If HTTP-date or unparsable, ignore and fall back
                pass
        # Fallback: parse from error.message e.g. "Please try again in 156ms" or "2s"
        try:
            msg = ((data or {}).get('error') or {}).get('message', '')
            import re
            m = re.search(r"in\s+(\d+)(ms|s)", msg)
            if m:
                val = float(m.group(1))
                unit = m.group(2)
                return val / 1000.0 if unit == 'ms' else val
        except Exception:
            pass
        return None

    def _post_with_retries(url, headers, json_body, proxies=None, max_retries=5, base_sleep=0.25, timeout=300):
        last_err_text = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=json_body, proxies=proxies, timeout=timeout)
                status = resp.status_code
                # Successful HTTP
                if 200 <= status < 300:
                    data = resp.json()
                    if isinstance(data, dict) and 'error' in data:
                        # Treat embedded error as failure (rare on 2xx)
                        last_err_text = str(data.get('error'))
                        raise requests.RequestException(last_err_text)
                    return data

                # Retryable errors
                if status in (429, 500, 502, 503, 504):
                    try:
                        data = resp.json()
                    except Exception:
                        data = None
                    sleep_s = _parse_retry_after_seconds(resp, data)
                    if sleep_s is None:
                        sleep_s = base_sleep * (2 ** attempt) * (1 + 0.2 * random.random())
                    try:
                        tqdm.write(f"⚠️ OpenAI HTTP {status}. Retrying in {sleep_s:.3f}s (attempt {attempt + 1}/3)", nolock=True)
                    except Exception:
                        pass
                    time.sleep(sleep_s)
                    last_err_text = resp.text[:500]
                    continue

                # Non-retryable
                last_err_text = resp.text[:500]
                resp.raise_for_status()
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err_text = str(e)
                sleep_s = base_sleep * (2 ** attempt) * (1 + 0.2 * random.random())
                try:
                    tqdm.write(f"⚠️ OpenAI network error. Retrying in {sleep_s:.3f}s (attempt {attempt + 1}/3): {last_err_text}", nolock=True)
                except Exception:
                    pass
                time.sleep(sleep_s)
                continue
            except requests.RequestException as e:
                # Non-retryable request error
                last_err_text = str(e)
                break
        raise Exception(f"Failed to call OpenAI API or parse response. Last error: {last_err_text}")

    resp_json = _post_with_retries(
        'https://api.openai.com/v1/chat/completions', headers, bodies, proxies=proxies, max_retries=3, timeout=None
    )

    # Parse successful response
    message = resp_json['choices'][0]['message']['content']
    usage = resp_json.get('usage', {})
    input_tokens = usage.get('prompt_tokens', 0)
    output_tokens = usage.get('completion_tokens', 0)
    prompt_tokens_details = usage.get('prompt_tokens_details') or {}
    cached_input_tokens = prompt_tokens_details.get('cached_tokens', 0) or 0
    # Log token stats via tqdm; don't change return signature
    total_tokens = (input_tokens or 0) + (output_tokens or 0)
    try:
        tqdm.write(f"📊 Tokens | prompt: {input_tokens} | cached: {cached_input_tokens} | completion: {output_tokens} | total: {total_tokens}", nolock=True)
    except Exception:
        pass
    logInfo = {"model": model, "prompt": prompt, "message": message}
    if delPrompt:
        del logInfo['prompt']
    logInfo.update(otherInfo)
    fileName = datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + "_" + str(uuid4()) + ".json"
    if logStart != "":
        fileName = logStart + "_" + fileName
    filePath = os.path.join(logPath, fileName)
    JS(filePath).newJS(logInfo)
    if return_token_counts:
        return message, input_tokens, output_tokens
    return message

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
