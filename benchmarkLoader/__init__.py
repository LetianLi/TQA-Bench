import os
from torch.utils.data import Dataset

import sys
sys.path.append('.')
from benchmarkUtils.database import DB

scaledRoot = 'dataset/scaledDB'

with open('benchmarkLoader/prompts/singleChoicePrompt.txt', 'r') as f:
    singlePrompt = f.read()

with open('benchmarkLoader/prompts/singleChoiceToolsPrompt.txt', 'r') as f:
    singleChoiceToolsPrompt = f.read()

with open('benchmarkLoader/prompts/multiChoicePrompt.txt', 'r') as f:
    multiPrompt = f.read()

with open('benchmarkLoader/prompts/batchedSingleChoicePrompt.txt', 'r') as f:
    batchedSinglePrompt = f.read()

# Stage-specific prompt construction functions for the agent
def build_stage1_prompt(question: str, available_tables: str) -> str:
    """
    Build the Stage 1 (Exploration) prompt.
    
    Args:
        question: The question to answer
        available_tables: String listing available tables
    
    Returns:
        Formatted prompt string for Stage 1
    """
    return f"""# Stage 1: Database Exploration

## Available Tools
- `peekTables` - Examine sample data from tables
- `readTables` - Read full table contents
- `executePython` - Run custom Python code

## Task
Explore the database to identify which tables and columns are relevant to answering the question.

## Available Tables
{available_tables}

## Question
{question}

## What You Need to Do
1. **Examine the tables** that seem relevant to the question using `peekTables` and `readTables`
2. **Identify the specific columns** that contain useful information for answering the question
3. **Understand the data structure** and relationships between tables
4. **Output your findings** in the exact format specified below

## Output Format
You must output your findings in this exact format:

```
TABLES RELEVANT: table1, table2, table3
TABLE table1 RELEVANT COLUMNS: col1, col2, col3
TABLE table2 RELEVANT COLUMNS: col1, col4, col5
TABLE table3 RELEVANT COLUMNS: col2, col6

NOTE TO NEXT STAGE: Brief summary of what you found relevant
```

## Important Notes
- **TABLES RELEVANT**: List only the table names that are actually relevant to answering the question
- **RELEVANT COLUMNS**: List only the columns that contain useful information for the question
- **NOTE TO NEXT STAGE**: This is a brief explanation of what you found and how it relates to the question. The next stage will use this information to understand what data to work with."""


def build_stage2_prompt(question: str, exploration_data: str, note_to_next: str) -> str:
    """
    Build the Stage 2 (Data Preparation) prompt.
    
    Args:
        question: The question to answer
        exploration_data: Relevant table information from Stage 1
        note_to_next: Note from Stage 1 explaining what was found relevant
    
    Returns:
        Formatted prompt string for Stage 2
    """
    return f"""# Stage 2: Data Preparation

## Available Tools
- `executePython` - Run custom Python code for data manipulation and verification

## Task
Create a combined table that contains all the information needed to answer the question.

## Question
{question}

## Exploration Results from Stage 1
{exploration_data}

## Stage 1's Note to You
{note_to_next}

## What You Need to Do
1. **Analyze the exploration results** to understand what data you have available
2. **Write Python code** that creates a combined table from the relevant tables (cleaned, filtered, and augmented)
   - **IMPORTANT**: Try to do all your data manipulation in a single executePython call
   - Each executePython call is independent - variables don't persist between calls
3. **Use `submitTable(dataframe, table_name)`** to store your new table
   - This is REQUIRED - it creates a table that Stage 3 will use to do final analysis on
   - Without using `submitTable`, Stage 3 won't have access to your prepared data
4. **Verify your generated table** using `executePython` to check its structure and content
5. **Output the table name** in the exact format specified below

## How the Tools Work Together
- **`executePython`**: Use this to write code that manipulates data and creates your combined table
- **`submitTable(dataframe, table_name)`**: This function provided in the executePython tool stores your DataFrame as a new table in the database
- **Important**: Stage 3 will only be able to access tables created with `submitTable`
- **Verification**: Use `executePython` after creating the table to verify it looks correct

## Output Format
You must output your results in this exact format:

```
PREPARED TABLE NAME: your_table_name

NOTE TO NEXT STAGE: Brief description of what the prepared table contains
```

## Important Notes
- **PREPARED TABLE NAME**: The exact name of the table you created with `submitTable`
- **NOTE TO NEXT STAGE**: Explain what your prepared table contains and how it helps answer the question
- **Table Creation**: You MUST use `submitTable` for Stage 3 to access your data
- **Verification**: Always verify your table with `executePython` before marking completion"""


def build_stage3_prompt(question: str, choices: str, prepared_table_data: str, note_to_next: str) -> str:
    """
    Build the Stage 3 (Analysis) prompt.
    
    Args:
        question: The question to answer
        choices: The multiple choice options
        prepared_table_data: Information about the prepared table from Stage 2
        note_to_next: Note from Stage 2 explaining what the prepared table contains
    
    Returns:
        Formatted prompt string for Stage 3
    """
    return f"""# Stage 3: Question Analysis

## Available Tools
- `executePython` - Run custom Python code for analysis and calculations

## Task
Analyze the prepared table to answer the question.

## Question
{question}

{choices}

## Prepared Table from Stage 2
{prepared_table_data}

## Stage 2's Note to You
{note_to_next}

## What You Need to Do
1. **Understand the prepared table** - examine its structure and content
2. **Use `executePython`** to perform any necessary calculations or analysis
3. **Analyze the data** to find the answer to the question
4. **Provide clear reasoning** for your conclusions
5. **Output your final answer** in the exact format specified below

## How to Use the Tools
- **`executePython`**: Use this to write Python code that analyzes the prepared table
- **Data Access**: The prepared table is already available in the database
- **Analysis**: Write code to filter, aggregate, or calculate what you need
- **Verification**: Use `executePython` to check intermediate results

## Output Format
You must output your final answer in this exact format:

```
Answer: A/B/C/D or N/A

I AM DONE
```

## Important Notes
- **Answer**: Choose the correct option (A, B, C, D) or N/A if none apply
- **I AM DONE**: This marks the completion of all stages
- **Reasoning**: Explain your analysis process before giving the final answer
- **Use the Data**: Base your answer on the prepared table, not external knowledge"""


class BenchmarkDataset(Dataset):
    def __init__(self, scale, markdown=True):
        self.scale = scale
        self.markdown = markdown
        self.dbRoot = os.path.join(scaledRoot, scale)
        self.maps = 'A B C D E'.split()

    def loadDB(self, dbn):
        dbp = os.path.join(self.dbRoot, dbn, f'{dbn}.sqlite')
        return DB(dbp)



