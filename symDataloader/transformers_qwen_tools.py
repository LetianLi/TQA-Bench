import sys
from typing import Any, Dict, List
from tqdm import tqdm
import threading
import time

# Add current directory to path
sys.path.append('.')

# Import required libraries
from smolagents import Tool, CodeAgent, TransformersModel
from llm_sandbox import SandboxSession
from benchmarkUtils.database import DatabaseObject
from benchmarkLoader import singleChoiceToolsPrompt
from symbolic import dataDict
from symDataloader.utils import TaskCore

# Global variables for session management
_SANDBOX_SESSION = None
_DB_OBJECT_INSTANCE = None
_SESSION_LOCK = threading.Lock()

class TableNamesTool(Tool):
    name = "get_table_names"
    description = "Retrieve all table names in the database"
    inputs = {}
    output_type = "string"
    
    def forward(self) -> str:
        global _DB_OBJECT_INSTANCE
        if _DB_OBJECT_INSTANCE is None:
            return "No database loaded."
        
        return f"The database contains {len(_DB_OBJECT_INSTANCE.tableNames)} tables: {', '.join(_DB_OBJECT_INSTANCE.tableNames)}."

class PeekTablesTool(Tool):
    name = "peek_tables"
    description = "Peek the first 5 rows of each of the given table names"
    inputs = {
        "table_names": {
            "type": "list",
            "description": "A list of table names to peek"
        }
    }
    output_type = "string"
    
    def forward(self, table_names: List[str]) -> str:
        global _DB_OBJECT_INSTANCE
        if _DB_OBJECT_INSTANCE is None:
            return "No database loaded."
        
        if not table_names:
            return "No table names provided."
        
        # Validate table names
        valid_tables = []
        invalid_tables = []
        for table_name in table_names:
            if table_name in _DB_OBJECT_INSTANCE.tableNames:
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
                table_data = _DB_OBJECT_INSTANCE.data[table_name]
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

class ReadTablesTool(Tool):
    name = "read_tables"
    description = "Read the given table names and return the entire data as a string"
    inputs = {
        "table_names": {
            "type": "list", 
            "description": "A list of table names to read"
        }
    }
    output_type = "string"
    
    def forward(self, table_names: List[str]) -> str:
        global _DB_OBJECT_INSTANCE
        if _DB_OBJECT_INSTANCE is None:
            return "No database loaded."
        
        # If no specific tables requested, read all
        if not table_names:
            table_names = _DB_OBJECT_INSTANCE.tableNames
        
        table_list = []
        for table_name in table_names:
            if table_name in _DB_OBJECT_INSTANCE.data:
                table_data = _DB_OBJECT_INSTANCE.data[table_name]
                table_str = ''
                for row in table_data:
                    table_str += ','.join(str(cell) for cell in row) + '\n'
                table_list.append(f'## {table_name}\n\n{table_str}')
        
        return '\n\n'.join(table_list)

class ExecutePythonTool(Tool):
    name = "execute_python"
    description = "Execute Python code in a sandboxed environment with access to the database through 'tables' variable"
    inputs = {
        "code": {
            "type": "string",
            "description": "Python code to execute. Use 'tables' variable to access database data."
        }
    }
    output_type = "string"
    
    def forward(self, code: str) -> str:
        global _SANDBOX_SESSION, _DB_OBJECT_INSTANCE
        
        if _SANDBOX_SESSION is None:
            return "No sandbox session available."
        
        if _DB_OBJECT_INSTANCE is None:
            return "No database loaded."
        
        try:
            # Prepare the database object in the sandbox
            setup_code = f"""
import sys
class DatabaseObject:
    def __init__(self, tableNames, data):
        self.tableNames = tableNames
        self.data = data

# Initialize tables object
tables = DatabaseObject(
    {_DB_OBJECT_INSTANCE.tableNames!r},
    {_DB_OBJECT_INSTANCE.data!r}
)
"""
            
            # Execute setup code first
            _SANDBOX_SESSION.run(setup_code)
            
            # Execute the user code
            result = _SANDBOX_SESSION.run(code)
            
            # Format the result
            output_parts = []
            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")
            if result.error:
                output_parts.append(f"ERROR:\n{result.error}")
            
            if not output_parts:
                output_parts.append("Code executed successfully with no output.")
            
            return '\n'.join(output_parts)
            
        except Exception as e:
            return f"ERROR: {str(e)}"

def initialize_sandbox():
    """Initialize the sandbox session"""
    global _SANDBOX_SESSION
    with _SESSION_LOCK:
        if _SANDBOX_SESSION is None:
            try:
                _SANDBOX_SESSION = SandboxSession(
                    lang="python",
                    backend="docker",
                    keep_template=True,
                    commit_container=True,
                    timeout=300  # 5 minutes timeout
                )
                _SANDBOX_SESSION.__enter__()
                
                # Install common packages
                _SANDBOX_SESSION.install([
                    "numpy", "pandas", "matplotlib", "seaborn", 
                    "scikit-learn", "scipy", "statistics"
                ])
                
                print("Sandbox session initialized successfully.")
                
            except Exception as e:
                print(f"Failed to initialize sandbox: {e}")
                _SANDBOX_SESSION = None

def cleanup_sandbox():
    """Cleanup the sandbox session"""
    global _SANDBOX_SESSION
    with _SESSION_LOCK:
        if _SANDBOX_SESSION is not None:
            try:
                _SANDBOX_SESSION.__exit__(None, None, None)
                print("Sandbox session cleaned up successfully.")
            except Exception as e:
                print(f"Error during sandbox cleanup: {e}")
            finally:
                _SANDBOX_SESSION = None

# Initialize tools
table_names_tool = TableNamesTool()
peek_tables_tool = PeekTablesTool()
read_tables_tool = ReadTablesTool()
execute_python_tool = ExecutePythonTool()

# Initialize model and agent
model = TransformersModel(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype="bfloat16"
)

agent = CodeAgent(
    tools=[table_names_tool, peek_tables_tool, read_tables_tool, execute_python_tool],
    model=model,
    max_steps=15,
    additional_authorized_imports=["numpy", "pandas", "matplotlib", "seaborn", "scipy", "statistics"]
)

# ---------------------------------------------------------------------------
# Prompt helper (adapted from lmstudio_qwen_tools.py)
# ---------------------------------------------------------------------------
def qaPrompt(dbStr, question, choices):
    totalQuestion = f'{question}\n\n{choices}'
    prompt = singleChoiceToolsPrompt.format(question=totalQuestion)
    return prompt

# ---------------------------------------------------------------------------
# Public API – plug into TaskCore
# ---------------------------------------------------------------------------
def qwenLocalCall(dbStr, question, choices):
    """
    Runs one inference via smolagents with transformers and returns the raw completion string.
    """
    global _DB_OBJECT_INSTANCE
    
    # Initialize sandbox if not already done
    initialize_sandbox()
    
    # Set up the database object
    _DB_OBJECT_INSTANCE = dbStr
    
    prompt = qaPrompt(dbStr, question, choices)
    
    # Create progress bars
    prompt_pbar = tqdm(total=100, desc="      Processing", position=3, leave=False)
    
    try:
        # Count input tokens (approximate)
        input_tokens = len(prompt.split()) * 1.3  # Rough approximation
        
        # Mark progress as started
        prompt_pbar.update(50)
        
        # Run the agent
        result = agent.run(prompt)
        
        # Mark progress as complete
        prompt_pbar.update(50)
        
        # Count output tokens (approximate)
        output_tokens = len(str(result).split()) * 1.3  # Rough approximation
        
        # Close progress bars
        prompt_pbar.close()
        
        response = str(result)
        return response.strip(), int(input_tokens), int(output_tokens)
        
    except Exception as e:
        prompt_pbar.close()
        error_msg = f"Error during agent execution: {str(e)}"
        print(error_msg)
        return error_msg, 0, 0

if __name__ == '__main__':
    try:
        dbRoot = 'symDataset/scaledDB' # path to extract symDataset.zip
        taskPath = 'symDataset/tasks/TableQA/dataset.sqlite' # TableQA's dataset.sqlite
        resultPath = 'symDataset/results/TableQA/transformers_qwen2.5_tools.sqlite' # result sqlite
        tc = TaskCore(dbRoot, taskPath, resultPath)
        
        for k in dataDict.keys():
            # for scale in ['8k', '16k', '32k', '64k']:
            for scale in ['8k']:
                timeSleep = 0
                # if scale == '16k':
                #     timeSleep = 30
                # elif scale == '32k':
                #     timeSleep = 60
                tc.testAll('Qwen2.5-7B-Instruct-Transformers-Tools', # The model name saved in taskPath
                        k, # dataset
                        scale, # 8k, 16k, 32k, 64k, 128k
                        False, # if use markdown
                        5, # dbLimit, 10 is ok
                        1, # sampleLimit, 1 is ok
                        14, # questionLimit, 14 is ok
                        qwenLocalCall,
                        timeSleep,
                        genDBObject=True)
    finally:
        # Cleanup sandbox when done
        cleanup_sandbox() 