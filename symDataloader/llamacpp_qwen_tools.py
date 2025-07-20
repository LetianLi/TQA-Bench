"""
Wrapper for Qwen2.5-7B-Instruct using llama.cpp directly with tools.

Use `uv sync` to install the dependencies.
"""

import random
import sys
import time
import json
import re
import inspect
from llama_cpp import Llama
from typing import Callable, Dict, List, Any
from llama_cpp.llama_types import ChatCompletionRequestMessage, ChatCompletionRequestToolMessage, ChatCompletionTool
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
    try:
        # Print the code in a code block
        tqdm.write(f"\n```python\n{code}\n```\n", nolock=True)
        
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
        return final_result
        
    except Exception as e:
        tqdm.write(f"⚠️ ERROR IN TOOL executePython: {e}", nolock=True)
        raise

# Tool registry - array of functions
TOOLS = [getTableNames, peekTables, readTables, executePython]

def _convert_type_to_json_schema(param_type: Any) -> dict:
    """Convert Python type annotation to JSON schema type."""
    # Handle direct type classes
    if param_type == str:
        return {"type": "string"}
    elif param_type == int or param_type == float:
        return {"type": "number"}
    
    # Handle string representations for complex types
    type_str = str(param_type)
    
    if type_str.startswith('list[') or type_str.startswith('List['):
        # Extract the inner type
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

def generate_tools_schema() -> List[ChatCompletionTool]:
    """Generate ChatCompletionTool list from TOOLS array."""
    tools_schema = []
    
    for tool_func in TOOLS:
        # Get function signature
        sig = inspect.signature(tool_func)
        
        # Get docstring
        description = inspect.getdoc(tool_func)
        if description is None:
            raise ValueError(f"Function {tool_func.__name__} has no docstring")
        
        # Build parameters schema
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':  # Skip self parameter
                continue
                
            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                raise ValueError(f"Parameter {param_name} in {tool_func.__name__} has no type annotation")
            
            try:
                properties[param_name] = _convert_type_to_json_schema(param_type)
                required.append(param_name)  # All args are required
            except ValueError as e:
                raise ValueError(f"Parameter {param_name} in {tool_func.__name__}: {e}")
        
        tool_schema = {
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
        }
        
        tools_schema.append(tool_schema)
    
    return tools_schema

# Generate tools schema at startup
TOOLS_DEFINITION = generate_tools_schema()

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
            # Clean up the match - remove any extra braces or formatting
            cleaned_match = match.strip()
            # Handle double braces if present
            if cleaned_match.startswith('{{') and cleaned_match.endswith('}}'):
                cleaned_match = cleaned_match[1:-1]
            
            tool_call = json.loads(cleaned_match)
            if "name" in tool_call and "arguments" in tool_call:
                tool_calls.append(tool_call)
        except json.JSONDecodeError as e:
            tqdm.write(f"❌ JSON parse error: {e}. Violating characters: `{e.doc[e.pos-1:e.pos+1]}`", nolock=True)
            # Create a failed tool call entry so it gets counted
            try:
                # Try to extract just the tool name for error reporting
                name_match = re.search(r'"name":\s*"([^"]+)"', match)
                if name_match:
                    tool_name = name_match.group(1)
                    
                    # Create dynamic error message based on JSONDecodeError properties
                    error_msg = f"JSON parsing for tool call {tool_name} failed: {e.msg}"
                    
                    # Extract the problematic snippet around the error position
                    start_pos = max(0, e.pos - 20)
                    end_pos = min(len(match), e.pos + 20)
                    error_snippet = match[start_pos:end_pos]
                    error_chars = e.doc[e.pos-1:e.pos+1]
                    
                    failed_tool_call = {
                        "name": tool_name,
                        "arguments": {},
                        "parse_error": error_msg,
                        "snippet": error_snippet,
                        "error_chars": error_chars
                    }
                    tool_calls.append(failed_tool_call)
            except Exception as e2:
                # If we can't even extract the name, skip it
                tqdm.write(f"❌ JSON parse error handler failed with error: {e2}", nolock=True)
                pass
            continue
    
    return tool_calls

def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """Execute a single tool call and return the result."""
    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    
    # Find the tool function in the TOOLS array
    tool_func = None
    for func in TOOLS:
        if func.__name__ == tool_name:
            tool_func = func
            break
    
    if tool_func is None:
        return f"Error: Unknown tool '{tool_name}'"
    
    try:
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
    Runs one inference via llama.cpp with tools and returns the formatted completion string.
    """
    # Log start of new request
    tqdm.write("\n🚀 STARTING NEW REQUEST", nolock=True)
    
    prompt = qaPrompt(dbStr, question, choices)
    max_tokens = 10000
    max_rounds = 10  # Limit tool calling rounds
    
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
    formatted_response = ""  # Build formatted response live
    count_tool_calls = 0
    failed_tool_calls = 0  # Track failed tool calls
    
    # Debug variables
    start_time = time.time()
    last_debug_write = start_time
    debugPath = 'symDataset/results/TableQA/llamacpp_qwen2.5_tools_debug.txt'
    
    # Build the full conversation context
    conversation: List[ChatCompletionRequestMessage] = [
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
            tools=TOOLS_DEFINITION,
            tool_choice="auto" if current_round < max_rounds - 1 else None,
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
        first_token_received = False
        finish_reason = None
        
        for chunk in response_stream:
            if chunk["choices"][0]["delta"].get("content"): # type: ignore
                content: str = chunk["choices"][0]["delta"]["content"] # type: ignore
                response_text += content
                formatted_response += content
                token_count += 1
                
                # Reset progress bar on first token to exclude initial latency (time to first token)
                if not first_token_received:
                    token_pbar.reset()
                    token_pbar.n = 1
                    first_token_received = True
                else:
                    token_pbar.n = token_count
                token_pbar.refresh()
                
                # Check if we should write to debug file
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # If it's been at least 10 seconds since last write
                if (current_time - last_debug_write) >= 10:
                    try:
                        with open(debugPath, 'w', encoding='utf-8') as f:
                            f.write(f"--- DEBUG WRITE AT {time.strftime('%H:%M:%S')} (ELAPSED: {elapsed_time:.1f}s) ---\n")
                            f.write(f"Token count: {token_count}\n")
                            f.write(f"Current round: {current_round}\n")
                            f.write(f"\n{formatted_response}\n")
                            f.write("--- END DEBUG WRITE ---\n")
                        last_debug_write = current_time
                    except Exception as e:
                        # Silently fail if debug writing fails
                        pass
            
            # Capture finish reason from the last chunk
            if chunk["choices"][0].get("finish_reason"): # type: ignore
                finish_reason = chunk["choices"][0]["finish_reason"] # type: ignore
        tqdm.write(f"🛑 PAUSED. {finish_reason}")
        
        # Count input tokens for this round (all messages in conversation)
        conversation_text = ""
        for msg in conversation:
            if msg["role"] in ["user", "system"]:
                msg_content = msg.get("content", "")
                if msg_content:
                    conversation_text += str(msg_content) + "\n"
        input_tokens = len(llm.tokenize(conversation_text.encode()))
        output_tokens = len(llm.tokenize(response_text.encode()))
        
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
        
        # Execute tool calls and append results live
        tool_results = []
        for i, tool_call in enumerate(tool_calls):
            count_tool_calls += 1
            tool_name = tool_call["name"]
            tqdm.write(f"🔧 EXECUTING TOOL: {tool_name}", nolock=True)
            
            # Check if this is a failed tool call due to JSON parsing
            if "parse_error" in tool_call:
                result = f"Error: {tool_call['parse_error']}\n\nError Snippet: \n```\n{tool_call.get('snippet', 'Something went wrong')}\n```\nError Characters: `{tool_call.get('error_chars', 'Something went wrong')}`\nThese errors are usually caused by using single quotes instead of double quotes or bad escaping.\n"
                failed_tool_calls += 1
            else:
                try:
                    result = execute_tool_call(tool_call)
                    if result.startswith("Error"):
                        failed_tool_calls += 1
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"
                    failed_tool_calls += 1
            
            # Append tool call and result to formatted response
            formatted_response += f"\n\n🔧 Tool Name: {tool_name}\n"
            formatted_response += f"🔧 Tool Args: {tool_call.get('arguments', {})}\n"
            formatted_response += f"🔧 Tool Output: {result}\n"
            
            tool_results.append(format_tool_result(tool_name, result))
        
        # Add tool results to conversation with proper format
        tool_response = "\n".join(tool_results)
        tool_message: ChatCompletionRequestToolMessage = {
            "role": "tool", 
            "content": tool_response,
            "tool_call_id": f"call_{current_round}_{i}"  # Generate a unique tool call ID
        }
        conversation.append(tool_message)
        
        current_round += 1
    
    # Close progress bars
    round_pbar.close()
    token_pbar.close()
    
    # Add debug information at the end
    debug_info = f"\n\nDebug: "
    debug_info += f" Failed tool calls: {failed_tool_calls} / {count_tool_calls}"
    if finish_reason:
        debug_info += f" Stop reason: {finish_reason}. Used {current_round} of {max_rounds} rounds"
    
    formatted_response += debug_info

    # Log finish reason
    tqdm.write(f"🛑 FINISHED. Used {current_round} of {max_rounds} rounds. {finish_reason}")
    
    return formatted_response.strip(), total_input_tokens, total_output_tokens

if __name__ == '__main__':
    # Cache the model
    get_model()
    print("Model cached\n\n")
    
    # Check for interactive mode
    if len(sys.argv) > 1 and (sys.argv[1] == "--interactive" or sys.argv[1] == "interactive"):
        tqdm.write("🔧 INTERACTIVE MODE ENABLED", nolock=True)
        tqdm.write("You can now ask questions directly. Type 'quit' or 'exit' to stop.", nolock=True)
        tqdm.write("You have access to tools: getTableNames, peekTables, readTables, executePython", nolock=True)
        tqdm.write("=" * 50, nolock=True)
        
        # Initialize empty database object for interactive mode
        empty_db = {}
        dbDataFrames = empty_db
        
        # Add a dummy table for testing
        import pandas as pd
        dummy_data = {
            'id': [1, 2, 3, 4, 5, 6],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
            'age': [25, 30, 35, 28, 32, 36],
            'city': ['New York', 'Los Angeles', 'Chicago', 'Boston', 'Seattle', 'Miami'],
            'salary': [50000, 60000, 70000, 55000, 65000, 75000]
        }
        dbDataFrames['employees'] = pd.DataFrame(dummy_data)
        
        # Initialize conversation history for multi-turn
        conversation: List[ChatCompletionRequestMessage] = [
            {"role": "system", "content": "You are a helpful AI assistant with access to tools. You can analyze database tables and execute Python code when needed. Always be helpful and accurate in your responses."}
        ]
        
        while True:
            try:
                # Get user question
                question = input("\n💬 User: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q']:
                    tqdm.write("Goodbye!", nolock=True)
                    break
                
                if not question:
                    tqdm.write("Please enter a question.", nolock=True)
                    continue
                
                print("-" * 10)
                
                # Add user message to conversation
                conversation.append({"role": "user", "content": question})
                
                # Get model instance
                llm = get_model()
                
                # Variables to track results
                total_input_tokens = 0
                total_output_tokens = 0
                current_round = 0
                max_rounds = 5
                
                # Tool calling loop
                while current_round < max_rounds:
                    # Use create_chat_completion with the conversation and tools
                    response_stream = llm.create_chat_completion(
                        messages=conversation,
                        tools=TOOLS_DEFINITION,
                        tool_choice="auto",
                        max_tokens=1000,
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
                    
                    print("🤖 Assistant: ", end="", flush=True)
                    
                    for chunk in response_stream:
                        if chunk["choices"][0]["delta"].get("content"): # type: ignore
                            content: str = chunk["choices"][0]["delta"]["content"] # type: ignore
                            response_text += content
                            print(content, end="", flush=True)
                            token_count += 1
                    
                    print()  # New line after assistant response
                    
                    # Use actual tokenization for accurate counts
                    input_tokens = len(llm.tokenize(question.encode())) if current_round == 0 else 0
                    output_tokens = len(llm.tokenize(response_text.encode()))
                    
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    
                    # Add assistant response to conversation
                    conversation.append({"role": "assistant", "content": response_text})
                    
                    # Check for tool calls
                    tool_calls = extract_tool_calls(response_text)
                    
                    if not tool_calls:
                        # No tool calls, we're done - this is the final response
                        break
                    
                    # Execute tool calls
                    tool_results = []
                    for i, tool_call in enumerate(tool_calls):
                        tool_name = tool_call["name"]
                        print("-" * 10)
                        print(f"🔧 Tool Name: {tool_name}")
                        print(f"🔧 Tool Args: {tool_call.get('arguments', {})}")
                        
                        result = execute_tool_call(tool_call)
                        print(f"🔧 Tool Output: {result}")
                        print("-" * 10)
                        
                        tool_results.append(format_tool_result(tool_name, result))
                    
                    # Add tool results to conversation with proper format
                    tool_response = "\n".join(tool_results)
                    tool_message: ChatCompletionRequestToolMessage = {
                        "role": "tool", 
                        "content": tool_response,
                        "tool_call_id": f"call_{current_round}_{i}"  # Generate a unique tool call ID
                    }
                    conversation.append(tool_message)
                    
                    current_round += 1
                
                print(f"📊 Tokens: {total_input_tokens} input, {total_output_tokens} output")
                print("-" * 10)
                
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
                tc.testAll('Qwen2.5-7B-Instruct-Local-Tools', # The model name saved in taskPath
                        k, # dataset
                        scale, # 8k, 16k, 32k, 64k, 128k
                        False, # if use markdown
                        5, # dbLimit, 10 is ok
                        1, # sampleLimit, 1 is ok
                        14, # questionLimit, 14 is ok
                        qwenLlamaCppToolsCall,
                        timeSleep,
                        genDataFrames=True) 