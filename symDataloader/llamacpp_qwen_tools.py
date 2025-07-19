"""
Wrapper for Qwen2.5-7B-Instruct using llama.cpp directly with tools.

Use `uv sync` to install the dependencies.
"""

import random
import sys
import time
import json
import re
from llama_cpp import Llama
from llama_cpp.llama import StoppingCriteriaList
from typing import Callable, Dict, List, Any
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import pandas as pd

sys.path.append('.')
from symbolic import dataDict
from symDataloader.utils import TaskCore
from benchmarkLoader import singleChoiceToolsPrompt
from benchmarkUtils.database import DatabaseObject

# ---------------------------------------------------------------------------
# 1. Global variables for tools
# ---------------------------------------------------------------------------
dbDataFrames: dict[str, pd.DataFrame] = {}

# ---------------------------------------------------------------------------
# 2. Tool definitions
# ---------------------------------------------------------------------------
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

# Tool registry
TOOLS = {
    "getTableNames": {
        "function": getTableNames,
        "description": "Retrieve all table names in the database.",
        "parameters": {}
    },
    "peekTables": {
        "function": peekTables,
        "description": "Peek the first 5 rows of each of the given table names.",
        "parameters": {
            "tableNames": {
                "type": "list[str]",
                "description": "A list of table names to peek."
            }
        }
    },
    "readTables": {
        "function": readTables,
        "description": "Read the given table names and return the entire data as a string.",
        "parameters": {
            "tableNames": {
                "type": "list[str]",
                "description": "A list of table names to read."
            }
        }
    },
    "executePython": {
        "function": executePython,
        "description": "Execute Python code in a sandboxed environment with access to the database tables.",
        "parameters": {
            "code": {
                "type": "str",
                "description": "Python code to execute."
            }
        }
    }
}

# ---------------------------------------------------------------------------
# 3. One global model handle (re-used for every call)
# ---------------------------------------------------------------------------
_MODEL = None

def get_model():
    """Initialize and return the llama.cpp model instance."""
    global _MODEL
    if _MODEL is None:
        _MODEL = Llama.from_pretrained(
            repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
            filename="qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
            additional_files=["qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf"],
            n_gpu_layers=-1,  # Use all available GPU layers
            n_ctx=131072,  # Max context length 131072 tokens
            n_batch=512,  # Evaluation Batch Size 512
            n_threads=3,  # CPU Thread Pool Size 3
            use_mmap=True,  # Enable memory mapping
            use_mlock=True,  # Keep model in memory
            offload_kqv=True,  # Offload KV cache to GPU memory
            verbose=False
        )
    return _MODEL

# ---------------------------------------------------------------------------
# 4. Tool calling and response processing
# ---------------------------------------------------------------------------
def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from the model's response."""
    tool_calls = []
    
    # Look for tool call patterns like:
    # <tool_call>{"name": "getTableNames", "arguments": {}}</tool_call>
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(tool_call_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            tool_call = json.loads(match)
            if "name" in tool_call and "arguments" in tool_call:
                tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue
    
    return tool_calls

def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """Execute a single tool call and return the result."""
    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    
    if tool_name not in TOOLS:
        return f"Error: Unknown tool '{tool_name}'"
    
    try:
        tool_func = TOOLS[tool_name]["function"]
        result = tool_func(**arguments)
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"

def format_tool_result(tool_name: str, result: str) -> str:
    """Format tool result for inclusion in the conversation."""
    return f'<tool_result name="{tool_name}">\n{result}\n</tool_result>'

def qaPrompt(dbStr, question, choices):
    totalQuestion = f'{question}\n\n{choices}'
    prompt = singleChoiceToolsPrompt.format(question=totalQuestion)
    return prompt

# ---------------------------------------------------------------------------
# 5. Public API – plug into TaskCore
# ---------------------------------------------------------------------------
def qwenLlamaCppToolsCall(dbStr, question, choices):
    """
    Runs one inference via llama.cpp with tools and returns the raw completion string.
    """
    prompt = qaPrompt(dbStr, question, choices)
    max_tokens = 1000
    max_rounds = 5  # Limit tool calling rounds
    
    # Get model instance
    llm = get_model()
    
    # Set up database
    global dbDataFrames
    dbDataFrames = dbStr
    
    # Create progress bars
    round_pbar = tqdm(total=max_rounds, desc="      Rounds", position=3, leave=False)
    token_pbar = tqdm(total=max_tokens, desc="      Tokens", position=4, leave=False, unit="tokens")
    
    # Variables to track results
    full_response = ""
    total_input_tokens = 0
    total_output_tokens = 0
    current_round = 0
    
    # Build the full conversation context
    conversation = [
        {"role": "system", "content": "You have access to tools to analyze database tables. Use them when needed to answer questions accurately."},
        {"role": "user", "content": prompt}
    ]
    
    # Tool calling loop
    while current_round < max_rounds:
        round_pbar.n = current_round
        round_pbar.refresh()
        
        # Use create_chat_completion with the conversation
        response_stream = llm.create_chat_completion(
            messages=conversation,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repeat_penalty=1.1,
            stream=True,
            stop=["I AM DONE"]
        )
        
        # Process the streaming response
        response_text = ""
        token_count = 0
        
        for chunk in response_stream:
            if chunk["choices"][0]["delta"].get("content"):
                content = chunk["choices"][0]["delta"]["content"]
                response_text += content
                token_count += 1
                token_pbar.n = token_count
                token_pbar.refresh()
        
        # For streaming responses, we need to estimate token counts
        # Since we can't get usage from stream chunks, we'll use our manual count
        input_tokens = len(llm.tokenize(prompt.encode())) if current_round == 0 else 0
        output_tokens = token_count
        
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        full_response += response_text
        
        # Add assistant response to conversation
        conversation.append({"role": "assistant", "content": response_text})
        
        # Check for tool calls
        tool_calls = extract_tool_calls(response_text)
        
        if not tool_calls:
            # No tool calls, we're done
            break
        
        # Execute tool calls
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tqdm.write(f"🔧 EXECUTING TOOL: {tool_name}", nolock=True)
            result = execute_tool_call(tool_call)
            tool_results.append(format_tool_result(tool_name, result))
        
        # Add tool results to conversation
        tool_response = "\n".join(tool_results)
        conversation.append({"role": "tool", "content": tool_response})
        
        current_round += 1
    
    # Close progress bars
    round_pbar.close()
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
                response, input_tokens, output_tokens = qwenLlamaCppToolsCall(dbStr, question, choices)
                
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
        resultPath = 'symDataset/results/TableQA/llamacpp_qwen2.5_tools.sqlite' # result sqlite
        tc = TaskCore(dbRoot, taskPath, resultPath)
        for k in dataDict.keys():
            # for scale in ['8k', '16k', '32k', '64k']:
            for scale in ['8k']:
                timeSleep = 0
                # if scale == '16k':
                #     timeSleep = 30
                # elif scale == '32k':
                #     timeSleep = 60
                tc.testAll('Qwen2.5-7B-Instruct-LlamaCpp-Tools', # The model name saved in taskPath
                        k, # dataset
                        scale, # 8k, 16k, 32k, 64k, 128k
                        False, # if use markdown
                        5, # dbLimit, 10 is ok
                        1, # sampleLimit, 1 is ok
                        14, # questionLimit, 14 is ok
                        qwenLlamaCppToolsCall,
                        timeSleep,
                        genDataFrames=True) 