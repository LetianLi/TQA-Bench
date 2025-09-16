"""
GPT-5-mini chat with tools for TableQA.

This mirrors the tool set and flow used in `llamacpp_qwen_tools.py` and
`lmstudio_qwen_tools.py`, but uses OpenAI's Chat Completions tools API.

Use `uv sync` to install the dependencies.
"""

import os
import sys
import json
import inspect
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Any, Dict, List
from tqdm import tqdm

sys.path.append('.')
from symbolic import dataDict
from symDataloader.utils import TaskCore
import symDataloader.testConfig as testConfig
from benchmarkLoader import singleChoiceToolsPrompt

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

# ---------------------------------------------------------------------------
# 1. Global variables for tools
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np

dbDataFrames: dict[str, pd.DataFrame] = {}


# ---------------------------------------------------------------------------
# 2. Tool definitions
# ---------------------------------------------------------------------------
def getTableNames():
    """
    Retrieve all table names in the database.
    """
    try:
        result = f"The database contains {len(dbDataFrames)} tables: {', '.join(dbDataFrames.keys())}."
        return result
    except Exception as e:
        tqdm.write(f"⚠️ ERROR IN TOOL getTableNames: {e}", nolock=True)
        raise


def peekTables(tableNames: list[str]):
    """
    Peek the first 5 rows of each of the given table names.

    Args:
        tableNames (list[str]): A list of table names to peek.
    """
    try:
        if not tableNames:
            result = "No table names provided."
            return result

        valid_tables = []
        invalid_tables = []
        for table_name in tableNames:
            if table_name in dbDataFrames:
                valid_tables.append(table_name)
            else:
                invalid_tables.append(table_name)

        result_parts = []

        if invalid_tables:
            result_parts.append(f"Invalid table names: {', '.join(invalid_tables)}")

        if valid_tables:
            result_parts.append(f"Peeking {len(valid_tables)} valid tables: {', '.join(valid_tables)}")

            for i, table_name in enumerate(valid_tables):
                df = dbDataFrames[table_name]
                total_rows = len(df)
                peek_rows = min(5, total_rows)

                result_parts.append(f"\nTable {i+1}/{len(valid_tables)}: {table_name}")
                result_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
                result_parts.append(f"Total rows: {total_rows}")
                result_parts.append(f"Peeking first {peek_rows} rows:")

                if total_rows == 0:
                    result_parts.append("  (Table is empty)")
                else:
                    for j in range(peek_rows):
                        row_data = df.iloc[j].tolist()
                        result_parts.append(f"  Row {j+1}: {', '.join(str(cell) for cell in row_data)}")
                    if total_rows > 5:
                        result_parts.append(f"  (Additional {total_rows - 5} rows not shown)")
                    else:
                        result_parts.append(f"  (All {total_rows} rows shown)")

        result = '\n'.join(result_parts)
        return result
    except Exception as e:
        tqdm.write(f"⚠️ ERROR IN TOOL peekTables: {e}", nolock=True)
        raise


def readTables(tableNames: list[str]):
    """
    Read the given table names and return the entire data as a string.
    If none provided or non-existent table provided, warn in return result.
    """
    try:
        if not tableNames:
            result = "No table names provided."
            return result

        valid_tables = []
        invalid_tables = []
        for tableName in tableNames:
            if tableName in dbDataFrames:
                valid_tables.append(tableName)
            else:
                invalid_tables.append(tableName)

        result_parts = []

        if invalid_tables:
            result_parts.append(f"Invalid table names: {', '.join(invalid_tables)}")

        if valid_tables:
            for tableName in valid_tables:
                df = dbDataFrames[tableName]
                tableStr = df.to_csv(index=False)
                result_parts.append(f'## {tableName}\n\n{tableStr}')
        else:
            if not result_parts:
                result_parts.append("No valid table names provided.")

        result = '\n\n'.join(result_parts)
        return result
    except Exception as e:
        tqdm.write(f"⚠️ ERROR IN TOOL readTables: {e}", nolock=True)
        raise


def executePython(code: str):
    """
    Execute the given Python code in a sandboxed environment.
    You may create variables, modify variables, modify the database through the tables variable, and read them back through the other tools.
    You must print the results to stdout in order to see them.
    A tables variable is already set up for you, so do not create a new variable by hardcoding values into the environment.
    You should never need to manually create a new variable for the database.

    Args:
        code (str): Python code to execute. Must be a valid Python string.

    Available Variables (These are prepopulated for you in the environment):
        - tables: A dictionary mapping table names to pandas DataFrames
        - tables[table_name]: pandas DataFrame for the specified table
        - tables.keys(): List of all table names in the database

    Available Libraries:
        - pandas: For data manipulation and analysis
        - numpy: For numerical computations
        - Python builtins: All standard Python built-in functions

    Note: You must use print() statements to output results. The output will be captured and returned.

    Security:
        - Only Python builtins, pandas, and numpy are available (no os, sys, subprocess, etc.)
        - No file system access
        - No network access
        - No ability to import dangerous modules
        - Execution times out after 2 minutes
    """
    try:
        tqdm.write(f"\n```python\n{code}\n```\n", nolock=True)

        # Check if last line contains a print statement, and if it doesn't, wrap the last line in a print statement
        code_lines = code.rstrip().split('\n')
        if code_lines:
            last_line = code_lines[-1].strip()
            if last_line and not last_line.startswith('print('):
                # Inject a notice print before the injected print
                code_lines[-1:]= [
                    f'_internal_executor_last_line = {last_line}',
                    'print("Printing evaluation of last line: " + str(_internal_executor_last_line))'
                ]
                code = '\n'.join(code_lines)

        import builtins
        import io
        import threading
        from contextlib import redirect_stdout, redirect_stderr

        # Create a sandboxed environment with only builtins
        safe_builtins = {}
        for name in dir(builtins):
            if not name.startswith('_'):
                safe_builtins[name] = getattr(builtins, name)

        # Add __import__ to allow import statements
        safe_builtins['__import__'] = builtins.__import__

        # Create a safe globals dict with the tables object, builtins, pandas, and numpy
        safe_globals = {
            '__builtins__': safe_builtins,
            'tables': dbDataFrames,
            'pd': pd,
            'pandas': pd,
            'np': np,
            'numpy': np
        }

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        error_occurred = False
        error_message = ""

        # Function to execute the code
        def execute_code():
            nonlocal error_occurred, error_message
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, safe_globals)
            except Exception as e:
                error_occurred = True
                error_message = str(e)

        # Execute code in a separate thread with timeout
        execution_thread = threading.Thread(target=execute_code)
        execution_thread.daemon = True
        execution_thread.start()
        execution_thread.join(timeout=120)

        if execution_thread.is_alive():
            error_occurred = True
            error_message = "Execution timed out after 2 minutes"

        # Get the captured output (regardless of whether exec succeeded or failed)
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Truncate each result part to a reasonable length (e.g., 1000 chars)
        def truncate(text, maxlen=1000):
            if not testConfig.limitContextGrowth:
                return text
            if text and len(text) > maxlen:
                return text[:maxlen] + f"\n... (truncated, {len(text)-maxlen} more chars. Output is truncated automatically at {maxlen} chars, either do more filtering, or if the result looks right, proceed.)"
            return text

        # Prepare the result
        result_parts = []
        if stdout_output:
            result_parts.append(f"STDOUT:\n{truncate(stdout_output)}")
        if stderr_output:
            result_parts.append(f"STDERR:\n{truncate(stderr_output)}")
        if error_occurred:
            result_parts.append(f"ERROR:\n{truncate(error_message)}")
        if not result_parts:
            result_parts.append("Code executed successfully with no output.")

        final_result = '\n'.join(result_parts)
        return final_result
    except Exception as e:
        tqdm.write(f"⚠️ ERROR IN TOOL executePython: {e}", nolock=True)
        raise


# Tool registry - array of functions
if testConfig.limitContextGrowth:
    TOOLS = [getTableNames, peekTables, executePython]
else:
    TOOLS = [getTableNames, peekTables, readTables, executePython]


def _convert_type_to_json_schema(param_type: Any) -> dict:
    """Convert Python type annotation to JSON schema type for OpenAI tools."""
    if param_type == str:
        return {"type": "string"}
    elif param_type == int or param_type == float:
        return {"type": "number"}

    type_str = str(param_type)
    if type_str.startswith('list[') or type_str.startswith('List['):
        start_idx = type_str.find('[') + 1
        end_idx = type_str.rfind(']')
        if start_idx > 0 and end_idx > start_idx:
            inner_type = type_str[start_idx:end_idx]
            if inner_type in ['str', 'string']:
                return {"type": "array", "items": {"type": "string"}}
            elif inner_type in ['int', 'float', 'number']:
                return {"type": "array", "items": {"type": "number"}}
            else:
                raise ValueError(f"Unsupported array type: {inner_type}")
        else:
            raise ValueError(f"Invalid list type format: {type_str}")
    else:
        raise ValueError(f"Unsupported type: {type_str}")


def generate_tools_schema() -> List[Dict[str, Any]]:
    """Generate OpenAI Chat Completions tools list from TOOLS array."""
    tools_schema: List[Dict[str, Any]] = []
    for tool_func in TOOLS:
        sig = inspect.signature(tool_func)
        description = inspect.getdoc(tool_func)
        if description is None:
            raise ValueError(f"Function {tool_func.__name__} has no docstring")

        properties: Dict[str, Any] = {}
        required: List[str] = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                raise ValueError(f"Parameter {param_name} in {tool_func.__name__} has no type annotation")
            properties[param_name] = _convert_type_to_json_schema(param_type)
            required.append(param_name)

        tools_schema.append({
            "type": "function",
            "function": {
                "name": tool_func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        })
    return tools_schema


TOOLS_DEFINITION = generate_tools_schema()


def format_tool_result(tool_name: str, result: str) -> str:
    """Format tool result for inclusion in the conversation display string."""
    return f'<tool_result name="{tool_name}">\n{result}\n</tool_result>'


def qaPrompt(dbStr, question, choices):
    totalQuestion = f'{question}\n\n{choices}'
    prompt = singleChoiceToolsPrompt.format(question=totalQuestion)
    return prompt


def _openai_chat(body: Dict[str, Any], proxies: Dict[str, str]) -> Dict[str, Any]:
    # Emulation mode: do not call API. Clear console, print request, and capture input as assistant response
    if USER_EMULATE:
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
        except Exception:
            pass
        print("[user-emulate] ChatCompletions request body:")
        try:
            print(json.dumps(body, indent=2, ensure_ascii=False))
        except Exception:
            print(str(body))
        print("\nEnter assistant response (single line). Options:")
        print("  - Plain text to return as assistant content")
        print("  - !call <toolName> <argsJson> to request a tool call (e.g., !call getTableNames {})")
        print("  - JSON object for a message, e.g., {\"content\": \"...\", \"tool_calls\": [...]} ")
        print("  - Full JSON response with choices[] if you prefer")
        user_resp = input("> ")

        # Helper to construct a response envelope expected by the caller
        def _wrap_message_as_response(message_obj: Dict[str, Any]) -> Dict[str, Any]:
            if 'role' not in message_obj:
                message_obj['role'] = 'assistant'
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": message_obj,
                        # finish_reason is informational for our debug footer only
                        "finish_reason": "tool_calls" if message_obj.get('tool_calls') else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        # 1) Shorthand: !call <toolName> <argsJson>. Example: `!call executePython {"code":"print(list(tables.keys()))"}`
        if user_resp.strip().startswith('!call'):
            try:
                parts = user_resp.strip().split(' ', 2)
                _, tool_name = parts[0], parts[1]
                args_json_str = parts[2] if len(parts) > 2 else "{}"
            except Exception:
                tool_name = ""
                args_json_str = "{}"
            message = {
                "content": "",
                "tool_calls": [
                    {
                        "id": "user_call_1",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": args_json_str,
                        },
                    }
                ],
            }
            return _wrap_message_as_response(message)

        # 2) Raw JSON input handling
        if user_resp.strip().startswith('{'):
            try:
                parsed = json.loads(user_resp)
                if 'choices' in parsed:
                    return parsed
                if 'message' in parsed:
                    return _wrap_message_as_response(parsed['message'])
                # Assume this is a message object
                return _wrap_message_as_response(parsed)
            except Exception:
                # Fallback to plain content
                pass

        # 3) Plain text content
        return _wrap_message_as_response({"content": user_resp})

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY is not set')

    session = requests.Session()
    retries = Retry(
        total=3,  # Total number of retries
        backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on rate limits and server errors
        allowed_methods=["POST"],  # Only retry on POST requests
        respect_retry_after_header=True,  # Respect Retry-After headers from OpenAI
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    resp = session.post('https://api.openai.com/v1/chat/completions', headers=headers, json=body, proxies=proxies)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# 3. Public API – plug into TaskCore
# ---------------------------------------------------------------------------
def gpt5miniToolsCall(dbStr, question, choices):
    """
    Runs one inference via OpenAI GPT-5-mini with tools and returns (message, input_tokens, output_tokens).
    """
    # Configuration
    model = 'gpt-5-mini'
    temperature = 1.0 # only supported value for gpt 5 mini
    top_p = None # not supported at all for gpt 5 mini
    max_rounds = 10

    proxies = {
        # Override via environment if needed; leave empty to use system env
        # 'http': 'socks5://127.0.0.1:1080',
        # 'https': 'socks5://127.0.0.1:1080',
    }

    # Prepare prompt and DB
    prompt = qaPrompt(dbStr, question, choices)
    global dbDataFrames
    dbDataFrames = dbStr

    # Conversation
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You have access to tools to analyze database tables. Use them when needed to answer questions accurately."},
        {"role": "user", "content": prompt},
    ]

    formatted_response = ""
    total_input_tokens = 0
    total_output_tokens = 0
    count_tool_calls = 0
    failed_tool_calls = 0

    current_round = 0
    finish_reason = None

    while current_round < max_rounds:
        body = {
            "model": model,
            "messages": messages,
            "tools": TOOLS_DEFINITION,
            "tool_choice": "auto",
            # "temperature": temperature,
            # "top_p": top_p,
        }

        try:
            resp_json = _openai_chat(body, proxies)
        except Exception as e:
            raise RuntimeError(f"Failed to call OpenAI API: {e}")

        choice = resp_json.get('choices', [{}])[0]
        message = choice.get('message', {})
        finish_reason = choice.get('finish_reason')
        usage = resp_json.get('usage', {})

        total_input_tokens += int(usage.get('prompt_tokens', 0) or 0)
        total_output_tokens += int(usage.get('completion_tokens', 0) or 0)

        content = message.get('content') or ""
        formatted_response += content

        tool_calls = message.get('tool_calls') or []

        # Add assistant message to conversation
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls if tool_calls else None
        })

        if not tool_calls:
            break

        # Execute tools
        tool_results_blocks: List[str] = []
        for tc in tool_calls:
            count_tool_calls += 1
            fn = tc.get('function', {})
            name = fn.get('name')
            arguments_str = fn.get('arguments') or "{}"
            try:
                args = json.loads(arguments_str)
            except Exception as e:
                args = {}
                failed_tool_calls += 1
                result = f"Error parsing arguments for {name}: {e}"
            else:
                # Find and execute
                tool_func = None
                for func in TOOLS:
                    if func.__name__ == name:
                        tool_func = func
                        break
                if tool_func is None:
                    result = f"Error: Unknown tool '{name}'"
                    failed_tool_calls += 1
                else:
                    try:
                        result = str(tool_func(**args))
                    except Exception as e:
                        result = f"Error executing {name}: {e}"
                        failed_tool_calls += 1

            # Append to formatted response for visibility
            formatted_response += f"\n\n🔧 Tool Name: {name}\n"
            formatted_response += f"🔧 Tool Args: {args}\n"
            formatted_response += f"🔧 Tool Output: {result}\n"

            # Add tool result message back to conversation
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.get('id'),
                "name": name,
                "content": result,
            }
            messages.append(tool_msg)

            tool_results_blocks.append(format_tool_result(name or "", result))

        # Optional: include a summarized tool_result block as a message (not required by API)
        formatted_response += f"\n"

        current_round += 1

    # Add debug footer
    debug_info = f"\n\nDebug:  Failed tool calls: {failed_tool_calls} / {count_tool_calls}"
    if finish_reason:
        debug_info += f"  Stop reason: {finish_reason}. Used {current_round} of {max_rounds} rounds"
    formatted_response += debug_info

    return formatted_response.strip(), total_input_tokens, total_output_tokens


if __name__ == '__main__':
    dbRoot = 'symDataset/scaledDB'  # path to extract symDataset.zip
    taskPath = 'symDataset/tasks/TableQA/dataset.sqlite'  # TableQA's dataset.sqlite
    resultPath = f'symDataset/results/TableQA/5_mini_tools{testConfig.saveFileSuffix}.sqlite'  # result sqlite
    tc = TaskCore(dbRoot, taskPath, resultPath)
    for k in dataDict.keys():
        # Apply table filter if specified
        if testConfig.tableFilter and k not in testConfig.tableFilter:
            continue
        
        for scale in testConfig.dbScales:
            timeSleep = 0
            tc.testAll('gpt-5-mini-tools',  # The model name saved in taskPath
                       k,  # dataset
                       scale,  # 8k, 16k, 32k, 64k, 128k
                       False,  # if use markdown
                       5,  # dbLimit, 10 is ok
                       1,  # sampleLimit, 1 is ok
                       14,  # questionLimit, 14 is ok
                       gpt5miniToolsCall,
                       timeSleep,
                       genDataFrames=True,
                       injectContextJunk=testConfig.injectContextJunk)
            
            # Clear console after each testAll call to reduce memory usage
            try:
                os.system('cls' if os.name == 'nt' else 'clear')
            except Exception:
                pass
    print("Done")
