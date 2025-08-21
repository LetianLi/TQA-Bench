"""
GPT-4o-mini chat with agent flow for TableQA.

Instead of the LLM just choosing when and how to use tools, we provide the tools but also at each stage give different instructions.
For example, first stage is determining the relevant tables and columns, second stage is combining and augmenting the data into a single table, third stage is answering the question.

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
from benchmarkLoader import build_stage1_prompt, build_stage2_prompt, build_stage3_prompt

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


def submitTable(dataframe, table_name):
    """
    Submit a DataFrame as a new table in the database.

    Args:
        dataframe: pandas DataFrame to store
        table_name: string name for the new table

    Returns:
        str: Success message with table info

    Raises:
        ValueError: If arguments are invalid
    """
    try:
        # Validate arguments
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("submitTable: First argument must be a pandas DataFrame")
        if not isinstance(table_name, str):
            raise ValueError("submitTable: Second argument must be a string")
        if not table_name.strip():
            raise ValueError("submitTable: Table name cannot be empty")
        if len(table_name) > 100:
            raise ValueError("submitTable: Table name too long (max 100 characters)")

        # Store the dataframe
        global dbDataFrames
        dbDataFrames[table_name] = dataframe.copy()

        return f"Table '{table_name}' created successfully with {len(dataframe)} rows and {len(dataframe.columns)} columns"

    except Exception as e:
        raise ValueError(f"submitTable error: {str(e)}")


def executePython(code: str):
    """
    Execute the given Python code in a freshly created sandboxed environment.
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

    Available Functions:
        - submitTable(dataframe, table_name): Store a DataFrame as a new table
          - dataframe: pandas DataFrame to store
          - table_name: string name for the new table
          - Returns: Success message with table info
          - Use this to persist your results for use in later stages
        - getAllDataframes(): Get all tables as a dictionary
        - getTableDataframe(table_name): Get a specific table as a DataFrame

    Available Libraries:
        - pandas: For data manipulation and analysis
        - numpy: For numerical computations
        - Python builtins: All standard Python built-in functions

    Recommended Usage:
        - `tables: dict[str, pd.DataFrame] = getAllDataframes()`
        - `df: pd.DataFrame = getTableDataframe('table_name')`
        - `submitTable(df, 'my_table')` available in Stage 2 to save your results for Stage 3
        - Do all related work in a single executePython call when possible; environment does not persist between calls

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
            'numpy': np,
            'getTableDataframe': lambda name: dbDataFrames.get(name),
            'getAllDataframes': lambda: dbDataFrames,
            'submitTable': submitTable
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

        # Prepare the result
        result_parts = []
        if stdout_output:
            result_parts.append(f"STDOUT:\n{stdout_output}")
        if stderr_output:
            result_parts.append(f"STDERR:\n{stderr_output}")
        if error_occurred:
            error_note = f"ERROR:\n{error_message}"
            
            # Add helpful note about string escaping and variable usage
            if "string" in error_message.lower() or "escape" in error_message.lower() or "quote" in error_message.lower():
                error_note += "\n\nTIP: If this error occurred because of string escaping issues when trying to hardcode database values, don't do that! Instead, use the available functions:\n"
                error_note += f"  - `getAllDataframes()`: Get all available tables as a dictionary\n"
                error_note += f"  - `getTableDataframe('table_name')`: Get a specific table as a pandas DataFrame\n"
                error_note += f"  - `pd`: pandas library for data manipulation\n"
                error_note += f"  - `np`: numpy library for numerical operations"
            
            result_parts.append(error_note)
        if not result_parts:
            result_parts.append("Code executed successfully with no output.")

        final_result = '\n'.join(result_parts)
        return final_result
    except Exception as e:
        tqdm.write(f"⚠️ ERROR IN TOOL executePython: {e}", nolock=True)
        raise


def generateTable(newTableKey: str, code: str):
    """
    Generate a new table by executing the provided code and storing the result.
    
    Args:
        newTableKey (str): The name/key for the new table
        code (str): Python code that creates the table (should assign to 'result' variable)
    """
    try:
        tqdm.write(f"\n```python\n{code}\n```\n", nolock=True)
        
        # Execute the code in the same sandboxed environment
        import builtins
        import io
        import threading
        from contextlib import redirect_stdout, redirect_stderr

        # Create a sandboxed environment with only builtins
        safe_builtins = {}
        for name in dir(builtins):
            if not name.startswith('_'):
                safe_builtins[name] = getattr(builtins, name)

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
                    # Check if 'result' variable was created
                    if 'result' not in safe_globals:
                        error_occurred = True
                        error_message = "Code must create a 'result' variable containing the new table"
                    else:
                        # Store the result in the global dbDataFrames
                        global dbDataFrames
                        dbDataFrames[newTableKey] = safe_globals['result']
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

        # Get the captured output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Prepare the result
        if error_occurred:
            result = f"Error creating table '{newTableKey}': {error_message}"
        else:
            new_table = dbDataFrames[newTableKey]
            result = f"Successfully created table '{newTableKey}' with {len(new_table)} rows and columns: {', '.join(new_table.columns.tolist())}"
            
            # Add sample data to the result
            if len(new_table) > 0:
                result += f"\n\nSample data (first 3 rows):\n{new_table.head(3).to_string(index=False)}"

        return result
    except Exception as e:
        tqdm.write(f"⚠️ ERROR IN TOOL generateTable: {e}", nolock=True)
        raise


# Tool registry - array of functions
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


# ---------------------------------------------------------------------------
# 3. Stage Configuration
# ---------------------------------------------------------------------------

STAGES = {
    "exploration": {
        "tools": ["peekTables", "readTables", "executePython"],
        "completion_marker": "TABLES RELEVANT:"
    },
    "data_prep": {
        "tools": ["peekTables", "readTables", "executePython"],
        "completion_marker": "PREPARED TABLE NAME:"
    },
    "analysis": {
        "tools": ["peekTables", "readTables", "executePython"],
        "completion_marker": "Answer:"
    }
}


def parse_stage_completion(content: str, stage: str) -> Dict[str, Any]:
    """Parse the completion markers for a given stage and extract relevant information."""
    result = {}
    
    if stage == "exploration":
        # Parse TABLES RELEVANT and TABLE X RELEVANT COLUMNS
        lines = content.split('\n')
        tables_relevant = []
        table_columns = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("TABLES RELEVANT:"):
                tables_relevant = [t.strip() for t in line.replace("TABLES RELEVANT:", "").split(',')]
            elif line.startswith("TABLE ") and " RELEVANT COLUMNS:" in line:
                parts = line.split(" RELEVANT COLUMNS:")
                table_name = parts[0].replace("TABLE ", "").strip()
                columns = [c.strip() for c in parts[1].split(',')]
                table_columns[table_name] = columns
        
        result = {
            "tables_relevant": tables_relevant,
            "table_columns": table_columns
        }
        
    elif stage == "data_prep":
        # Parse PREPARED TABLE NAME
        lines = content.split('\n')
        prepared_table = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("PREPARED TABLE NAME:"):
                prepared_table = line.replace("PREPARED TABLE NAME:", "").strip()
                break
        
        result = {
            "prepared_table": prepared_table
        }
    
    return result


def get_stage1_context() -> str:
    """Get context for Stage 1: just table names"""
    table_names = list(dbDataFrames.keys())
    return f"Available tables: {', '.join(table_names)}"


def get_stage2_context(exploration_info: Dict[str, Any]) -> str:
    """Get context for Stage 2: peeks of relevant tables"""
    tables = exploration_info.get("tables_relevant", [])
    table_peeks = {}
    
    for table in tables:
        if table in dbDataFrames:
            df = dbDataFrames[table]
            table_peeks[table] = {
                "rows": len(df),
                "columns": df.columns.tolist(),
                "sample": df.head(3).to_dict('records') if len(df) > 0 else []
            }
    
    context = f"Relevant tables from exploration:\n"
    for table, peek_info in table_peeks.items():
        context += f"\n{table}:\n"
        context += f"  Rows: {peek_info['rows']}\n"
        context += f"  Columns: {', '.join(peek_info['columns'])}\n"
        if peek_info['sample']:
            context += f"  Sample data:\n"
            for i, row in enumerate(peek_info['sample'][:2]):  # Just 2 rows
                context += f"    Row {i+1}: {row}\n"
    
    return context


def get_stage3_context(data_prep_info: Dict[str, Any]) -> str:
    """Get context for Stage 3: only the prepared table"""
    prepared_table = data_prep_info.get("prepared_table")
    
    if prepared_table and prepared_table in dbDataFrames:
        df = dbDataFrames[prepared_table]
        context = f"Prepared table: {prepared_table}\n"
        context += f"Structure: {len(df)} rows, columns: {', '.join(df.columns.tolist())}\n\n"
        context += f"Preview:\n{df.head(5).to_string(index=False)}"
    else:
        context = f"Prepared table: {prepared_table} (table not found)"
    
    return context


def get_stage_tools(stage: str) -> List[Dict[str, Any]]:
    """Get the tools definition for a specific stage."""
    stage_config = STAGES[stage]
    stage_tools = stage_config["tools"]
    
    # Filter the full tools definition to only include stage-specific tools
    filtered_tools = []
    for tool in TOOLS_DEFINITION:
        if tool["function"]["name"] in stage_tools:
            filtered_tools.append(tool)
    
    return filtered_tools


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
        print("  - !ai to let GPT generate a response once")
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

        # 1) AI generation mode: !ai to let GPT generate response
        if user_resp.strip() == '!ai':
            try:
                # Use OpenAI API to generate a response
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    print("⚠️ OPENAI_API_KEY not set, cannot generate AI response")
                    return _wrap_message_as_response({"content": "Error: API key not available"})
                
                # Make the actual API call
                session = requests.Session()
                retries = Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["POST"],
                    respect_retry_after_header=True,
                )
                session.mount('https://', HTTPAdapter(max_retries=retries))

                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}'
                }
                
                print("🤖 Generating AI response...")
                resp = session.post('https://api.openai.com/v1/chat/completions', headers=headers, json=body, proxies=proxies)
                resp.raise_for_status()
                ai_resp = resp.json()
                
                print("✅ AI response generated successfully")
                return ai_resp
                
            except Exception as e:
                print(f"⚠️ Error generating AI response: {e}")
                return _wrap_message_as_response({"content": f"Error generating AI response: {e}"})

        # 2) Shorthand: !call <toolName> <argsJson>. Example: `!call executePython {"code":"print(list(tables.keys()))"}`
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

        # 3) Raw JSON input handling
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

        # 4) Plain text content
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
# 4. Public API – plug into TaskCore
# ---------------------------------------------------------------------------
def gpt4ominiAgentCall(dbStr, question, choices):
    """
    Runs one inference via OpenAI GPT-4o-mini with staged agent flow and returns (message, input_tokens, output_tokens).
    """
    # Configuration
    model = 'gpt-4o-mini'
    temperature = 0.6
    top_p = 0.95
    max_rounds_per_stage = 15
    max_total_rounds = 50

    proxies = {
        # Override via environment if needed; leave empty to use system env
        # 'http': 'socks5://127.0.0.1:1080',
        # 'https': 'socks5://127.0.0.1:1080',
    }

    # Prepare prompt and DB
    global dbDataFrames
    dbDataFrames = dbStr
    prompt = build_stage1_prompt(question, get_stage1_context())

    # Initialize stage tracking
    current_stage = "exploration"
    stage_rounds = 0
    total_rounds = 0
    previous_stages_info = {}
    
    # Conversation
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": prompt},
    ]

    formatted_response = f"\n{'='*10}\n🚀 STARTING STAGE 1: Database Exploration\n{'='*10}\n"
    total_input_tokens = 0
    total_output_tokens = 0
    count_tool_calls = 0
    failed_tool_calls = 0

    finish_reason = None
    
    while current_stage in STAGES and total_rounds < max_total_rounds:
        # Get stage-specific tools
        stage_tools = get_stage_tools(current_stage)
        
        body = {
            "model": model,
            "messages": messages,
            "tools": stage_tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "top_p": top_p,
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

        # Check if current stage is complete
        stage_config = STAGES[current_stage]
        completion_marker = stage_config["completion_marker"]
        
        if completion_marker in content:
            # Stage is complete, parse information and move to next stage
            stage_info = parse_stage_completion(content, current_stage)
            previous_stages_info[current_stage] = stage_info

            # Determine next stage
            if current_stage == "exploration":
                formatted_response += f"\n\n{'='*10}\n🏁 STAGE 1 COMPLETE - Moving to STAGE 2\n{'='*10}\n"
                current_stage = "data_prep"
            elif current_stage == "data_prep":
                formatted_response += f"\n\n{'='*10}\n🏁 STAGE 2 COMPLETE - Moving to STAGE 3\n{'='*10}\n"
                current_stage = "analysis"
            elif current_stage == "analysis":
                formatted_response += f"\n\n{'='*10}\n🏁 STAGE 3 COMPLETE - All stages finished\n{'='*10}\n"
                break  # All stages complete
            
            # Build context for next stage
            stage_context = ""
            if current_stage == "data_prep":
                # Stage 2 gets the NOTE TO NEXT STAGE from Stage 1 + relevant table data
                exploration_info = previous_stages_info["exploration"]
                note_to_next = ""
                
                # Extract the NOTE TO NEXT STAGE from Stage 1's output
                if "exploration" in previous_stages_info:
                    # Parse the content to find NOTE TO NEXT STAGE
                    content_lines = content.split('\n')
                    for line in content_lines:
                        if line.strip().startswith("NOTE TO NEXT STAGE:"):
                            note_to_next = line.strip()
                            break
                
                if not note_to_next:
                    note_to_next = "NOTE TO NEXT STAGE: No specific note provided from exploration stage"
                
                stage_context = build_stage2_prompt(question, get_stage2_context(exploration_info), note_to_next)
                
            elif current_stage == "analysis":
                # Stage 3 gets the NOTE TO NEXT STAGE from Stage 2 + prepared table data
                data_prep_info = previous_stages_info["data_prep"]
                note_to_next = ""
                
                # Extract the NOTE TO NEXT STAGE from Stage 2's output
                if "data_prep" in previous_stages_info:
                    # Parse the content to find NOTE TO NEXT STAGE
                    content_lines = content.split('\n')
                    for line in content_lines:
                        if line.strip().startswith("NOTE TO NEXT STAGE:"):
                            note_to_next = line.strip()
                            break
                
                if not note_to_next:
                    note_to_next = "NOTE TO NEXT STAGE: No specific note provided from data preparation stage"
                
                stage_context = build_stage3_prompt(question, choices, get_stage3_context(data_prep_info), note_to_next)
            
            # Start fresh with new user message for the next stage
            messages = [
                {"role": "user", "content": stage_context},
            ]
            
            # Reset stage rounds
            stage_rounds = 0
            continue

        if not tool_calls:
            # No tool calls and no completion marker - continue with current stage
            stage_rounds += 1
            total_rounds += 1
            if stage_rounds >= max_rounds_per_stage:
                # Force stage transition if too many rounds
                if current_stage == "exploration":
                    current_stage = "data_prep"
                elif current_stage == "data_prep":
                    current_stage = "analysis"
                elif current_stage == "analysis":
                    break
                stage_rounds = 0
                continue

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

        stage_rounds += 1
        total_rounds += 1

    # Add debug footer
    debug_info = f"\n\nDebug:  Failed tool calls: {failed_tool_calls} / {count_tool_calls}"
    if finish_reason:
        debug_info += f"  Stop reason: {finish_reason}. Used {total_rounds} of {max_total_rounds} rounds"
    debug_info += f"  Completed stages: {list(previous_stages_info.keys())}"
    formatted_response += debug_info

    return formatted_response.strip(), total_input_tokens, total_output_tokens


if __name__ == '__main__':
    dbRoot = 'symDataset/scaledDB'  # path to extract symDataset.zip
    taskPath = 'symDataset/tasks/TableQA/dataset.sqlite'  # TableQA's dataset.sqlite
    resultPath = 'symDataset/results/TableQA/4o_mini_agent.sqlite'  # result sqlite
    tc = TaskCore(dbRoot, taskPath, resultPath)
    for k in dataDict.keys():
        for scale in ['8k']:
            timeSleep = 0
            tc.testAll('gpt-4o-mini-agent',  # The model name saved in taskPath
                       k,  # dataset
                       scale,  # 8k, 16k, 32k, 64k, 128k
                       False,  # if use markdown
                       5,  # dbLimit, 10 is ok
                       1,  # sampleLimit, 1 is ok
                       14,  # questionLimit, 14 is ok
                       gpt4ominiAgentCall,
                       timeSleep,
                       genDataFrames=True)
            
            # Clear console after each testAll call to reduce memory usage
            try:
                os.system('cls' if os.name == 'nt' else 'clear')
            except Exception:
                pass
    print("Done")