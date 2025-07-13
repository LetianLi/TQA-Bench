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
from benchmarkLoader import singleChoiceToolsPrompt
from benchmarkUtils.database import DatabaseObject

# Tools
dbObjectInstance: DatabaseObject = DatabaseObject([], {})
def getTableNames():
    """
    Retrieve all table names in the database.
    """
    return f"The database contains {len(dbObjectInstance.tableNames)} tables: {', '.join(dbObjectInstance.tableNames)}."

def peekTables(tableNames: list[str]):
    """
    Peek the first 5 rows of each of the given table names.
    
    Args:
        tableNames (list[str]): A list of table names to peek.
    """
    if not tableNames:
        return "No table names provided."
    
    # Validate table names
    valid_tables = []
    invalid_tables = []
    for table_name in tableNames:
        if table_name in dbObjectInstance.tableNames:
            valid_tables.append(table_name)
        else:
            invalid_tables.append(table_name)
    
    result_parts = []
    
    # Report invalid tables
    if invalid_tables:
        result_parts.append(f"Invalid table names: {', '.join(invalid_tables)}")
    
    # Peek valid tables
    if valid_tables:
        result_parts.append(f"Peeking {len(valid_tables)} valid tables: {', '.join(valid_tables)}")
        
        for i, table_name in enumerate(valid_tables):
            table_data = dbObjectInstance.data[table_name]
            column_names = table_data[0]  # First row contains column names
            data_rows = table_data[1:]    # Remaining rows are data
            
            total_rows = len(data_rows)
            peek_rows = min(5, total_rows)
            
            result_parts.append(f"\nTable {i+1}/{len(valid_tables)}: {table_name}")
            result_parts.append(f"Columns: {', '.join(column_names)}")
            result_parts.append(f"Total rows: {total_rows}")
            result_parts.append(f"Peeking first {peek_rows} rows:")
            
            if total_rows == 0:
                result_parts.append("  (Table is empty)")
            else:
                for j in range(peek_rows):
                    row_data = data_rows[j]
                    result_parts.append(f"  Row {j+1}: {', '.join(str(cell) for cell in row_data)}")
                if total_rows > 5:
                    result_parts.append(f"  (Additional {total_rows - 5} rows not shown)")
                else:
                    result_parts.append(f"  (All {total_rows} rows shown)")
    
    return '\n'.join(result_parts)

def readTables(tableNames: list[str]):
    """
    Read the given table names and return the entire data as a string.
    """
    tableList = []
    for tableName, tableData in dbObjectInstance.data.items():
        tableStr = ''
        for row in range(len(tableData)):
            tableStr += ','.join(str(cell) for cell in tableData[row]) + '\n'
        tableList.append(f'## {tableName}\n\n{tableStr}')
    return '\n\n'.join(tableList)

def executePython(code: str):
    """
    Execute the given Python code in a sandboxed environment.
    You may create variables, modify variables, modify the database through the tables variable, and read them back through the other tools.
    
    Args:
        code (str): Python code to execute. Must be a valid Python string.
        
    Available Variables:
        - tables: DatabaseObject instance with tableNames and data properties
        - tables.tableNames: List of all table names in the database
        - tables.data: Dictionary mapping table names to 2D arrays where:
          * tables.data[table_name][0] = list of column names
          * tables.data[table_name][1:] = data rows (each row is a list of values)
        
    Security:
        - Only Python builtins are available (no os, sys, subprocess, etc.)
        - No file system access
        - No network access
        - No ability to import dangerous modules
        - Execution times out after 2 minutes
    """
    import builtins
    import sys
    import io
    import signal
    import threading
    import time
    from contextlib import redirect_stdout, redirect_stderr
    
    # Create a sandboxed environment with only builtins
    safe_builtins = {}
    for name in dir(builtins):
        if not name.startswith('_'):
            safe_builtins[name] = getattr(builtins, name)
    
    # Create a safe globals dict with only the tables object and builtins
    safe_globals = {
        '__builtins__': safe_builtins,
        'tables': dbObjectInstance
    }
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    error_occurred = False
    error_message = ""
    timeout_occurred = False
    
    # Function to execute the code
    def execute_code():
        nonlocal error_occurred, error_message
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code in the sandboxed environment
                exec(code, safe_globals)
        except Exception as e:
            # Even if exec crashes, we still want to capture any output that was printed
            error_occurred = True
            error_message = str(e)
    
    # Execute code in a separate thread with timeout
    execution_thread = threading.Thread(target=execute_code)
    execution_thread.daemon = True
    execution_thread.start()
    
    # Wait for execution to complete or timeout after 2 minutes (120 seconds)
    execution_thread.join(timeout=120)
    
    if execution_thread.is_alive():
        timeout_occurred = True
        error_occurred = True
        error_message = "Execution timed out after 2 minutes"
    
    # Get the captured output (regardless of whether exec succeeded or failed)
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()
    
    # Prepare the result
    result_parts = []
    if stdout_output:
        result_parts.append(f"STDOUT:\n{stdout_output}")
    if stderr_output:
        result_parts.append(f"STDERR:\n{stderr_output}")
    
    # Add error information if an exception occurred
    if error_occurred:
        result_parts.append(f"ERROR:\n{error_message}")
    
    if not result_parts:
        result_parts.append("Code executed successfully with no output.")
    
    try:
        return '\n'.join(result_parts)
    finally:
        stdout_capture.close()
        stderr_capture.close()

# ---------------------------------------------------------------------------
# 1. One global model handle (re-used for every call)
# ---------------------------------------------------------------------------
lms.configure_default_client("localhost:5841")
# _MODEL = lms.llm("qwen/qwen3-8b")   # alias used by LM Studio catalog
# _MODEL = lms.llm("qwen/qwen2.5-7b-instruct")   # alias used by LM Studio catalog
_MODEL = lms.llm("qwen/qwen2.5-7b-instruct")   # alias used by LM Studio catalog

# ---------------------------------------------------------------------------
# 2. Prompt helper (same as before)
# ---------------------------------------------------------------------------
def qaPrompt(dbStr, question, choices):
    totalQuestion = f'{question}\n\n{choices}'
    prompt = singleChoiceToolsPrompt.format(question=totalQuestion)
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
    
    # Variables to track results across rounds
    full_response = ""
    total_input_tokens = 0
    total_output_tokens = 0
    current_round = 0
    token_count = 0
    
    # Debug variables
    import time
    start_time = time.time()
    last_debug_write = start_time
    tokenStream = ""
    debugPath = 'symDataset/results/TableQA/lmstudio_qwen2.5_tools_debug.txt'
    
    # Create progress bars
    prompt_pbar = tqdm(total=100, desc="      Prompt (Round 0)", position=3, leave=False)
    token_pbar = tqdm(total=maxTokens, desc="      Tokens (Round 0)", position=4, leave=False, unit="tokens")

    # Track prompt processing progress
    def on_prompt_progress(progress, round_index):
        prompt_pbar.n = int(progress * 100)
        prompt_pbar.refresh()

    def on_prediction_fragment(fragment, round_index):
        nonlocal token_count, tokenStream, last_debug_write
        token_count += 1
        token_pbar.n = token_count
        token_pbar.refresh()
        
        # Add fragment to tokenStream
        if hasattr(fragment, 'content'):
            tokenStream += fragment.content
        
        # Check if we should write to debug file
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # If it's been at least 10 seconds since last write
        if (current_time - last_debug_write) >= 10:
            try:
                with open(debugPath, 'w', encoding='utf-8') as f:
                    f.write(f"--- DEBUG WRITE AT {time.strftime('%H:%M:%S')} (ELAPSED: {elapsed_time:.1f}s) ---\n")
                    f.write(f"Token count: {token_count}\n")
                    f.write(f"Current round: {round_index}\n")
                    f.write(f"Token stream so far:\n{tokenStream}\n")
                    f.write("--- END DEBUG WRITE ---\n")
                last_debug_write = current_time
            except Exception as e:
                # Silently fail if debug writing fails
                pass

    def on_message(message):
        nonlocal full_response, tokenStream
        # Extract content from assistant response messages
        if hasattr(message, 'content') and message.content:
            # Handle Sequence[TextData | FileHandle | ToolCallRequestData]
            for content_item in message.content:
                if hasattr(content_item, 'type'):
                    if content_item.type == "text":
                        full_response += content_item.text
                    elif content_item.type == "toolCallRequest":
                        tool_request_str = "\n" + str(content_item) + "\n"
                        full_response += tool_request_str
                        tokenStream += tool_request_str
                    elif content_item.type == "toolCallResult":
                        # Add tool call results to the response
                        tool_result_str = "\n" + str(content_item) + "\n"
                        full_response += tool_result_str
                        tokenStream += tool_result_str
                    else:
                        other_content_str = "\n" + str(content_item) + "\n"
                        full_response += other_content_str
                        tokenStream += other_content_str
                else:
                    # Fallback for unexpected content types
                    fallback_str = "\n" + str(content_item) + "\n"
                    full_response += fallback_str
                    tokenStream += fallback_str

    def on_prediction_completed(round_result):
        nonlocal total_input_tokens, total_output_tokens, current_round, full_response
        current_round += 1
        
        # Aggregate token counts
        if hasattr(round_result, 'prompt_tokens_count'):
            total_input_tokens += round_result.prompt_tokens_count
        if hasattr(round_result, 'predicted_tokens_count'):
            total_output_tokens += round_result.predicted_tokens_count
        
        # Reset progress bars for new round
        prompt_pbar.reset()
        token_pbar.reset()
        prompt_pbar.set_description(f"      Prompt (Round {current_round})")
        token_pbar.set_description(f"      Tokens (Round {current_round})")
        prompt_pbar.refresh()
        token_pbar.refresh()

    def on_round_end(round_index):
        nonlocal full_response
        # Add round end separator to response for debugging
        full_response += f"\n\n--- ROUND {round_index + 1} ENDED (TOOLS RESOLVED) ---\n\n"

    def on_round_start(round_index):
        nonlocal token_count, full_response
        token_count = 0
        
        # Add round separator to response for debugging
        if round_index > 0:  # Don't add separator before first round
            full_response += f"\n\n--- STARTING ROUND {round_index + 1} ---\n\n"
        
        prompt_pbar.reset()
        token_pbar.reset()
        prompt_pbar.set_description(f"      Prompt (Round {round_index + 1})")
        token_pbar.set_description(f"      Tokens (Round {round_index + 1})")
        prompt_pbar.refresh()
        token_pbar.refresh()
    
    def handle_invalid_tool_request(error, request):
        nonlocal full_response
        full_response += f"\n\n--- INVALID TOOL REQUEST ---\n\n"
        full_response += f"Error: {error}\n"
        full_response += f"Request: {request}\n"
        if request is None:
            return full_response + "\nUnrecoverable error"
        return None

    # Stream the completion
    global dbObjectInstance
    dbObjectInstance = dbStr

    act_result = _MODEL.act(
        prompt,
        [getTableNames, peekTables, readTables, executePython],
        config={ # Best parameters according to https://huggingface.co/Qwen/Qwen3-8B#best-practices
            "maxTokens": maxTokens,
            "temperature": 0.6,
            "topPSampling": 0.95,
            "topKSampling": 20,
            "minPSampling": 0,
            "repeatPenalty": 1.1,
            "stopStrings": ["I AM DONE"]
        },
        on_prompt_processing_progress=on_prompt_progress,
        on_prediction_fragment=on_prediction_fragment,
        on_message=on_message,
        on_prediction_completed=on_prediction_completed,
        on_round_start=on_round_start,
        on_round_end=on_round_end,
        handle_invalid_tool_request=handle_invalid_tool_request
    )

    # Close progress bars
    prompt_pbar.close()
    token_pbar.close()

    return full_response.strip(), total_input_tokens, total_output_tokens
    

if __name__ == '__main__':
    dbRoot = 'symDataset/scaledDB' # path to extract symDataset.zip
    taskPath = 'symDataset/tasks/TableQA/dataset.sqlite' # TableQA's dataset.sqlite
    resultPath = 'symDataset/results/TableQA/lmstudio_qwen2.5_tools.sqlite' # result sqlite
    debugPath = 'symDataset/results/TableQA/lmstudio_qwen2.5_tools_debug.txt' # debug file
    tc = TaskCore(dbRoot, taskPath, resultPath)
    for k in dataDict.keys():
        # for scale in ['8k', '16k', '32k', '64k']:
        for scale in ['8k']:
            timeSleep = 0
            # if scale == '16k':
            #     timeSleep = 30
            # elif scale == '32k':
            #     timeSleep = 60
            tc.testAll('Qwen2.5-7B-Instruct-Local-Tools', # The model name saved in taskPath
                    k, # dataset
                    scale, # 8k, 16k, 32k, 64k, 128k
                    False, # if use markdown
                    5, # dbLimit, 10 is ok
                    1, # sampleLimit, 1 is ok
                    14, # questionLimit, 14 is ok
                    qwenLocalCall,
                    timeSleep,
                    genDBObject=True)
