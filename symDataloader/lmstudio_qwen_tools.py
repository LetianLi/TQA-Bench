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
import pandas as pd
dbDataFrames: dict[str, pd.DataFrame] = {}

def getTableNames():
    """
    Retrieve all table names in the database.
    """
    tqdm.write("🔧 ENTERING TOOL: getTableNames", nolock=True)
    try:
        result = f"The database contains {len(dbDataFrames)} tables: {', '.join(dbDataFrames.keys())}."
        tqdm.write("🔧 EXITING TOOL: getTableNames", nolock=True)
        return result
    except Exception as e:
        tqdm.write(f"🔧 ERROR IN TOOL getTableNames: {e}", nolock=True)
        raise

def peekTables(tableNames: list[str]):
    """
    Peek the first 5 rows of each of the given table names.
    
    Args:
        tableNames (list[str]): A list of table names to peek.
    """
    tqdm.write("🔧 ENTERING TOOL: peekTables", nolock=True)
    try:
        if not tableNames:
            result = "No table names provided."
            tqdm.write("🔧 EXITING TOOL: peekTables", nolock=True)
            return result
        
        # Validate table names
        valid_tables = []
        invalid_tables = []
        for table_name in tableNames:
            if table_name in dbDataFrames:
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
        tqdm.write("🔧 EXITING TOOL: peekTables", nolock=True)
        return result
    except Exception as e:
        tqdm.write(f"🔧 ERROR IN TOOL peekTables: {e}", nolock=True)
        raise

def readTables(tableNames: list[str]):
    """
    Read the given table names and return the entire data as a string.
    """
    tqdm.write("🔧 ENTERING TOOL: readTables", nolock=True)
    try:
        tableList = []
        for tableName, df in dbDataFrames.items():
            tableStr = df.to_csv(index=False)
            tableList.append(f'## {tableName}\n\n{tableStr}')
        result = '\n\n'.join(tableList)
        tqdm.write("🔧 EXITING TOOL: readTables", nolock=True)
        return result
    except Exception as e:
        tqdm.write(f"🔧 ERROR IN TOOL readTables: {e}", nolock=True)
        raise

def executePython(code: str):
    """
    Execute the given Python code in a sandboxed environment.
    You may create variables, modify variables, modify the database through the tables variable, and read them back through the other tools.
    
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
        
    Examples:
        # Get table names and basic info
        table_names = list(tables.keys())
        if table_names:
            table_name = table_names[0]
            df = tables[table_name]
            print(f"Table: {table_name}")
            print(f"Columns: {list(df.columns)}")
            print(f"Shape: {df.shape}")
            print(f"Row count: {len(df)}")
            
            # Print the last row of data
            if len(df) > 0:
                last_row = df.iloc[-1].tolist()
                print(f"Last row: {last_row}")
            
            # Basic pandas operations
            print(f"Data types: {df.dtypes}")
            print(f"Summary statistics:")
            print(df.describe())
            
            # Filter and analyze data
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                print(f"Numeric columns: {list(numeric_cols)}")
                print(f"Mean of first numeric column: {df[numeric_cols[0]].mean()}")
    
    Note: You must use print() statements to output results. The output will be captured and returned.
        
    Security:
        - Only Python builtins, pandas, and numpy are available (no os, sys, subprocess, etc.)
        - No file system access
        - No network access
        - No ability to import dangerous modules
        - Execution times out after 2 minutes
    """
    tqdm.write("🔧 ENTERING TOOL: executePython", nolock=True)
    try:
        import builtins
        import sys
        import io
        import signal
        import threading
        import time
        import pandas as pd
        import numpy as np
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
        
        final_result = '\n'.join(result_parts)
        tqdm.write("🔧 EXITING TOOL: executePython", nolock=True)
        return final_result
        
    except Exception as e:
        tqdm.write(f"🔧 ERROR IN TOOL executePython: {e}", nolock=True)
        raise

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
                        tqdm.write("🔧 TOOL CALL REQUEST DETECTED", nolock=True)
                        tool_request_str = "\n" + str(content_item) + "\n"
                        full_response += tool_request_str
                        tokenStream += tool_request_str
                    elif content_item.type == "toolCallResult":
                        tqdm.write("🔧 TOOL CALL RESULT RECEIVED", nolock=True)
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
        
        # Aggregate token counts from round_result.stats
        if hasattr(round_result, 'stats'):
            if hasattr(round_result.stats, 'prompt_tokens_count'):
                total_input_tokens += round_result.stats.prompt_tokens_count
            if hasattr(round_result.stats, 'predicted_tokens_count'):
                total_output_tokens += round_result.stats.predicted_tokens_count
        
        # Reset progress bars for new round
        prompt_pbar.reset()
        token_pbar.reset()
        prompt_pbar.set_description(f"      Prompt (Round {current_round})")
        token_pbar.set_description(f"      Tokens (Round {current_round})")
        prompt_pbar.refresh()
        token_pbar.refresh()

    def on_round_end(round_index):
        nonlocal full_response
        tqdm.write(f"🔧 ROUND {round_index + 1} ENDED (TOOLS RESOLVED)", nolock=True)
        # Add round end separator to response for debugging
        full_response += f"\n\n--- ROUND {round_index + 1} ENDED (TOOLS RESOLVED) ---\n\n"

    def on_round_start(round_index):
        nonlocal token_count, full_response
        token_count = 0
        
        if round_index > 0:  # Don't log for first round
            tqdm.write(f"🔧 STARTING ROUND {round_index + 1}", nolock=True)
        
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
        tqdm.write(f"🔧 INVALID TOOL REQUEST DETECTED: {error}", nolock=True)
        full_response += f"\n\n--- INVALID TOOL REQUEST ---\n\n"
        full_response += f"Error: {error}\n"
        full_response += f"Request: {request}\n"
        if request is None:
            return full_response + "\nUnrecoverable error"
        return None

    # Stream the completion
    global dbDataFrames
    dbDataFrames = dbStr

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
    # Check for interactive mode
    if len(sys.argv) > 1 and (sys.argv[1] == "--interactive" or sys.argv[1] == "interactive"):
        tqdm.write("🔧 INTERACTIVE MODE ENABLED", nolock=True)
        tqdm.write("You can now ask questions directly. Type 'quit' or 'exit' to stop.", nolock=True)
        tqdm.write("=" * 50, nolock=True)
        
        # Initialize empty database object for interactive mode
        empty_db = {}
        
        while True:
            try:
                # Get user question
                question = input("\nYour question: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q']:
                    tqdm.write("Goodbye!", nolock=True)
                    break
                
                if not question:
                    tqdm.write("Please enter a question.", nolock=True)
                    continue
                
                tqdm.write(f"\nProcessing: {question}", nolock=True)
                tqdm.write("-" * 50, nolock=True)
                
                # Set up interactive parameters
                dbStr = empty_db
                choices = "This is a user question, you do not need to format as a multiple choice question, just answer the query directly."
                
                # Call the model
                response, input_tokens, output_tokens = qwenLocalCall(dbStr, question, choices)
                
                tqdm.write("\nResponse:", nolock=True)
                tqdm.write("=" * 50, nolock=True)
                tqdm.write(response, nolock=True)
                tqdm.write("=" * 50, nolock=True)
                tqdm.write(f"Tokens: {input_tokens} input, {output_tokens} output", nolock=True)
                
            except KeyboardInterrupt:
                tqdm.write("\nInterrupted by user. Goodbye!", nolock=True)
                break
            except Exception as e:
                tqdm.write(f"Error: {e}", nolock=True)
                continue
    
    else:
        # Default mode - run the original benchmark
        dbRoot = 'symDataset/scaledDB' # path to extract symDataset.zip
        taskPath = 'symDataset/tasks/TableQA/dataset.sqlite' # TableQA's dataset.sqlite
        resultPath = 'symDataset/results/TableQA/lmstudio_qwen2.5_tools_v2.sqlite' # result sqlite
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
                        genDataFrames=True)
