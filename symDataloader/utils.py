import os
import re
import sqlite3
import time
from tqdm import tqdm
import pandas as pd

import sys

sys.path.append(".")
from benchmarkUtils.database import DB


def extractAnswer(text: str) -> str:
    patt = r"answer: *([A-F]+|N/A)"
    grps = re.findall(patt, text, re.IGNORECASE)
    if grps:
        return grps[-1].upper()
    return ""

def testValid(rightIdx, choice_a, choice_b, choice_c, choice_d):
    """
    Private function to test if a question is valid.
    
    Args:
        rightIdx: Index of the correct answer (0-3)
        choice_a, choice_b, choice_c, choice_d: The answer choices
        
    Returns:
        tuple: (is_valid, reason) where reason is None if valid, or a string describing why invalid
    """
    # Check if rightIdx is valid
    if rightIdx is None or not (0 <= rightIdx <= 3):
        return False, "invalid rightIdx"
    
    # Get all choices and check if they're all "nan"
    choices = [choice_a, choice_b, choice_c, choice_d]
    
    # Check if all choices are "nan" (case-insensitive)
    all_nan = all(
        choice is None or 
        str(choice).lower() in ["nan", "none", ""] or 
        (isinstance(choice, str) and choice.strip() == "")
        for choice in choices
    )
    
    if all_nan:
        return False, "all nan"
    
    # Check if the correct choice exists and is valid
    if rightIdx < len(choices):
        correct_choice = choices[rightIdx]
        if (correct_choice is None or 
            str(correct_choice).lower() in ["nan", "none", "unknown"] or
            (isinstance(correct_choice, str) and correct_choice.strip() == "")):
            return False, "none answer"
    
    return True, None

class TaskCore:
    choicesMap = "A B C D E F".split()
    createresulttemplate = """
    create table if not exists {table_name} (
        model text,
        scale text,
        markdown integer,
        dbidx integer,
        sampleidx integer,
        questionidx integer,
        gt text,
        pred text,
        correct integer,
        error text,
        message text,
        input_tokens integer,
        output_tokens integer,
        qtype text,
        validQuestion integer,
        primary key (model, scale, markdown, dbidx, sampleidx, questionidx)
    );
    """

    primarykeycheck = """
    select 1
    from {table_name}
    where model = ? and scale = ? and markdown = ? and dbidx = ? and sampleidx = ? and questionidx = ?;
    """

    inserttemplate = """
    insert or ignore into {table_name}
    (model, scale, markdown, dbidx, sampleidx, questionidx, gt, pred, correct, error, message, input_tokens, output_tokens, qtype, validQuestion)
    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    def __init__(self, dbRoot, taskPath, resultPath) -> None:
        self.dbRoot = dbRoot
        self.taskPath = taskPath
        self.resultPath = resultPath
        dirPath = os.path.dirname(self.resultPath)
        os.makedirs(dirPath, exist_ok=True)

        self.taskConn = sqlite3.connect(self.taskPath)
        self.taskCur = self.taskConn.cursor()
        self.resultConn = sqlite3.connect(self.resultPath)
        self.resultCur = self.resultConn.cursor()

        self.tableNames = TaskCore.getAllTableNames(self.taskCur)

        for tn in self.tableNames:
            self.resultCur.execute(TaskCore.createresulttemplate.format(table_name=tn))
            
            # Check if validQuestion column exists, add it if it doesn't
            try:
                self.resultCur.execute(f"PRAGMA table_info({tn})")
                columns = [col[1] for col in self.resultCur.fetchall()]
                if 'validQuestion' not in columns:
                    self.resultCur.execute(f"ALTER TABLE {tn} ADD COLUMN validQuestion INTEGER DEFAULT 0")
                    print(f"Added validQuestion column to existing table {tn}")
            except Exception as e:
                print(f"Warning: Could not add validQuestion column to table {tn}: {e}")
                
        self.resultConn.commit()

    def loadTaskItem(self, dbn, scale, dbIdx, sampleIdx, questionIdx):
        self.taskCur.execute(
            "SELECT * FROM {dbn} WHERE scale=? AND dbIdx=? AND sampleIdx=? AND questionIdx=?;".format(
                dbn=dbn
            ),
            (scale, dbIdx, sampleIdx, questionIdx),
        )
        item = self.taskCur.fetchone()
        if item:
            return item
        return None

    @staticmethod
    def getAllTableNames(cursor: sqlite3.Cursor):
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tableNames = []
        items = cursor.fetchall()
        if items:
            for it in items:
                tableNames.append(it[0])
        return tableNames

    @staticmethod
    def getTableColumns(cursor: sqlite3.Cursor, tbn: str):
        cursor.execute("SELECT * FROM {table_name} LIMIT 1;".format(table_name=tbn))
        return [tup[0] for tup in cursor.description]

    @staticmethod
    def generateChoices(choicesList: list):
        choices = []
        for i in range(len(choicesList)):
            choices.append(f"{TaskCore.choicesMap[i]}) {choicesList[i]}")
        return "\n".join(choices)

    @staticmethod
    def getRightChoices(rightIdx: int):
        rightChoices = []
        idxStr = str(rightIdx)
        for char in idxStr:
            val = int(char)
            rightChoices.append(TaskCore.choicesMap[val])
        rightChoices.sort()
        return "".join(rightChoices)

    def resultCheck(self, dbn, model, scale, markdown, dbIdx, sampleIdx, questionIdx):
        """
        return: True means this index have already tested.
        """
        self.resultCur.execute(
            TaskCore.primarykeycheck.format(table_name=dbn),
            (model, scale, markdown, dbIdx, sampleIdx, questionIdx),
        )
        if self.resultCur.fetchone():
            return True
        return False

    @staticmethod
    def tableLlamaSerialize(tbn: str, df: pd.DataFrame):
        cols = df.columns.to_list()
        colStr = "| " + " | ".join([str(it) for it in cols]) + " |"
        sz = len(df)
        rows = []
        for i in range(sz):
            row = df.iloc[i].to_list()
            row = [str(it) for it in row]
            rows.append("| " + " | ".join(row) + " |")
        rowsStr = " [SEP] ".join(rows)
        totalStr = f"[TLE] The table title is {tbn} . [TAB] {colStr} [SEP] {rowsStr}"
        return totalStr

    def testAll(
        self,
        model,
        dbn,
        scale,
        markdown,
        dbLimit,
        sampleLimit,
        questionLimit,
        func,
        timeSleep=0,
        genDataFrames=False,
    ):
        """
        func need to be a call function have 3 arguments -- dbStr, question, choicesStr
        """
        for dbIdx in tqdm(range(dbLimit), desc=f"DBs ({dbn})", position=0, leave=True):
            for sampleIdx in tqdm(range(sampleLimit), desc=f"  Sample", position=1, leave=False):
                for questionIdx in tqdm(range(questionLimit), desc=f"    Questions", position=2, leave=False):
                    # if self.resultCheck(
                    #     dbn, model, scale, markdown, dbIdx, sampleIdx, questionIdx
                    # ):
                    #     continue
                    item = self.loadTaskItem(dbn, scale, dbIdx, sampleIdx, questionIdx)
                    if item is None:
                        continue
                    dbp = os.path.join(self.dbRoot, scale, dbn, f"{dbIdx}.sqlite")
                    db = DB(dbp)
                    dbStr = ""
                    if genDataFrames:
                        # Create pandas DataFrames from the database tables
                        dbStr = db.tables  # This is already a dict of DataFrames
                    elif markdown is None:
                        dbStrList = []
                        for tbn, df in db.tables.items():
                            dbStrList.append(TaskCore.tableLlamaSerialize(tbn, df))
                        dbStr = " ".join(dbStrList)
                    else:
                        dbStr = f"#{dbn}\n\n{db.defaultSerialization(markdown)}"
                    choicesStr = TaskCore.generateChoices(item[-4:])
                    gt = TaskCore.getRightChoices(item[-5])
                    question = item[-6]
                    qtype = item[-7]

                    # Check if the question is valid
                    rightIdx = item[-5]  # rightIdx is the 5th element from the end
                    choice_a, choice_b, choice_c, choice_d = item[-4:]  # Last 4 elements are the choices
                    is_valid, reason = testValid(rightIdx, choice_a, choice_b, choice_c, choice_d)
                    
                    pred = ""
                    error = ""
                    res = ""
                    resStr = ""
                    input_tokens = 0
                    output_tokens = 0
                    
                    # Only process the question if it's valid
                    if is_valid:
                        try:
                            res = func(dbStr, question, choicesStr)
                            if isinstance(res, tuple) and len(res) == 3: # Check if we return token info
                                resStr, input_tokens, output_tokens = res
                            else:
                                resStr = str(res)  # Ensure res is a string
                                # If no token info provided, set to 0
                                input_tokens = 0
                                output_tokens = 0
                            
                            pred = extractAnswer(resStr)
                            
                            # Write debug output to lastOutput.txt
                            debug_file_path = os.path.join(os.path.dirname(self.resultPath), "lastOutput.txt")
                            with open(debug_file_path, 'w', encoding='utf-8') as f:
                                f.write(resStr + "\n\n" + "*"*20 + "\n\nExtracted answer: " + pred)
                            
                            time.sleep(timeSleep)
                        except Exception as e:
                            print(e)
                            error = str(e)
                    else:
                        # For invalid questions, set empty values but still insert
                        pred = ""
                        resStr = ""
                        error = f"Invalid question: {reason}"
                    
                    self.resultCur.execute(
                        TaskCore.inserttemplate.format(table_name=dbn),
                        (
                            model,
                            scale,
                            markdown,
                            dbIdx,
                            sampleIdx,
                            questionIdx,
                            gt,
                            pred,
                            gt == pred,
                            error,
                            resStr,
                            input_tokens,
                            output_tokens,
                            qtype,
                            1 if is_valid else 0,  # validQuestion column
                        ),
                    )
                    self.resultConn.commit()


if __name__ == "__main__":
    data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
    df = pd.DataFrame(data)
    res = TaskCore.tableLlamaSerialize("tbn", df)
    print(res)
