import os
import pandas as pd
import sqlite3
from tqdm import tqdm
from pprint import pprint

class DatabaseObject:
    """
    A Python object representing a database with tableNames and tables properties.
    tableNames: List of table names in the database
    tables: Dictionary mapping table names to 2D arrays where table[0] is column names and table[n>=1] are data rows
    """
    def __init__(self, table_names: list[str], tables_dict: dict[str, list[list[str]]]):
        self.tableNames = table_names
        self.data = tables_dict

class DB:
    def __init__(self, dbp, initTables=True):
        """
        This remains compatible with older code, so `initTables` is added to initialize tables.
        If you are only performing sampling, set this to `False` to speed things up.
        """
        self.dbp = dbp
        self.dbn = dbp.split('/')[-1].split('.')[0]

        self.conn = sqlite3.connect(self.dbp)
        # Trade some safety for significant performance gain
        # self.conn.execute('PRAGMA journal_mode=WAL;')
        # self.conn.commit()
        self.conn.execute('PRAGMA synchronous=OFF;')
        self.conn.commit()
        self.conn.execute('PRAGMA cache_size=-16777216') # Set 4 GB cache size
        self.conn.commit()

        self.curs = self.conn.cursor()
        self.tableNames = []
        self.tableNames = self.getAllTableNames()

        self.tables = {}
        if initTables:
            self.tables = self.initDataFrame()

    def defaultSerialization(self, markdown=False, dbObject=False):
        """
        markdown: If True, serialize as a Markdown table; if False, serialize as CSV.
        dbObject: If True, return a Python object with tableNames and tables properties.
        This is the default serialization strategy; you can choose whether to use Markdown or dbObject.
        If options are passed, dbObject takes precedence, then markdown, then csv by default.
        """
        tables = self.initDataFrame()

        if dbObject:
            # Convert DataFrames to 2D arrays where table[0] is column names and table[n>=1] are rows
            tables_dict = {}
            for table_name, df in tables.items():
                # Get column names as the first row
                column_names = df.columns.tolist()
                # Get all data rows
                data_rows = df.values.tolist()
                # Combine column names and data rows
                table_array = [column_names] + data_rows
                tables_dict[table_name] = table_array
            
            return DatabaseObject(list(tables.keys()), tables_dict)

        tableList = []
        for k, v in tables.items():
            if markdown:
                tableList.append(f'## {k}\n\n{v.to_markdown(index=False)}')
            else:
                tableList.append(f'## {k}\n\n{v.to_csv(index=False)}')
        return '\n\n'.join(tableList)

    def rowCount(self, tabName):
        # Get row count
        self.curs.execute(f'SELECT COUNT(*) FROM [{tabName}];')
        return self.curs.fetchall()[0][0]

    def schema(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        schema = cursor.fetchall()
        schemaList = [s[0] for s in schema]
        cursor.close()
        return '\n'.join(schemaList)
    
    def initDataFrame(self):
        """
        Note: Read tables through this interface so that any whitespace in table names is replaced with “_”.
        """
        if len(self.tables) > 0:
            return self.tables
        tablesName = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';", self.conn)

        dataframes = {}
        for tn in tablesName['name']:
            newTN = tn.strip().replace(' ', '_').replace('-', '_').replace('\t', '_') # Make sure to replace spaces, tabs, etc. in the DataFrame name; otherwise the code won’t run
            dataframes[newTN] = pd.read_sql(f"SELECT * FROM [{tn}]", self.conn)
        self.tables = dataframes
        return dataframes
    
    def getAllTableNames(self):
        if len(self.tableNames) != 0:
            return self.tableNames
        # Get all table names
        self.curs.execute("""SELECT name 
                    FROM sqlite_master 
                    WHERE type = 'table' 
                    AND name NOT LIKE 'sqlite_%';
                """)
        res = self.curs.fetchall()
        tableNames = [item[0] for item in res]
        self.tableNames = tableNames
        return tableNames

    def getAllColumnNames(self, tbn):
        # Get all column names for tbn
        self.curs.execute(f"PRAGMA table_info([{tbn}]);")
        res = self.curs.fetchall()
        columnNames = [item[1] for item in res]
        return columnNames

    def getSingleForeignKey(self, tbn):
        # Get foreign key information for tbn
        self.curs.execute(f"PRAGMA foreign_key_list([{tbn}]);")
        res = self.curs.fetchall()
        foreignKey = []
        for item in res:
            foreignKey.append({'currentColumn': item[3], 'foreignTable': item[2], 'foreignColumn': item[4]})
            foreignKey[-1]['foreignColumn'] = self.getTableKey(foreignKey[-1]['foreignTable'])[0] if foreignKey[-1]['foreignColumn'] is None else foreignKey[-1]['foreignColumn']
        return foreignKey

    def getAllForeignKeys(self):
        # The foreign‑key relationships obtained here are used only for topological sorting!
        tableNames = self.getAllTableNames()
        allForeignKeys = {}
        for tbn in tableNames:
            ret = self.getSingleForeignKey(tbn)
            allForeignKeys[tbn] = ret
        return allForeignKeys

    def getTableKey(self, tbn):
        """
        Get primary key
        """
        self.curs.execute(f'PRAGMA table_info([{tbn}]);')
        res = self.curs.fetchall()
        k = []
        for item in res:
            if item[5] == 1:
                k.append(item[1])
        return k

    def getAllRootKeys(self):
        """
        `getAllForeignKeys` returns a dict whose keys are table names (`tbn`) and whose values list
        the relationships where `currentColumn` in `tbn` connects to `foreignColumn` in `foreignTable`.
        For our purposes we need, for each root table, to know which linkedTable.linkedColumn points
        into its rootColumn; the same column can be referenced by multiple tables.
        """
        allForeignKeys = self.getAllForeignKeys()
        allRootKeys = {}
        for k in allForeignKeys.keys():
            allRootKeys[k] = []
        for k, v in allForeignKeys.items():
            for item in v:
                # Note: It may be omitted
                allRootKeys[item['foreignTable']].append({'rootColumn': self.getTableKey(item['foreignTable'])[0] if item['foreignColumn'] is None else item['foreignColumn'],
                                                          'linkedTable': k,
                                                          'linkedColumn': item['currentColumn']})
        for k, v in allRootKeys.items():
            for item in v:
                for ik, iv in item.items():
                    if iv == None:
                        print(f'{ik} {iv}')
        return allRootKeys

    def getTopology(self):
        allForeignKeys = self.getAllForeignKeys()
        tableRely = {}
        for k, v in allForeignKeys.items():
            tableRely[k] = [item['foreignTable'] for item in v]

        topoOrder = []
        currTable = None
        while len(tableRely) > 0:
            currTable = None
            for k, v in tableRely.items():
                if len(v) == 0:
                    topoOrder.append(k)
                    currTable = k
                    break
            if currTable is None:
                print('There exists loop in this database.')
                return []
            for k, v in tableRely.items():
                while topoOrder[-1] in v:
                    v.remove(topoOrder[-1])
            del tableRely[currTable]
        return topoOrder

    def sample(self, dstPath, sampleNum=16, removeEmpty=True, removeOldVer=True):
        # Note: re‑initialize the cursor and connection each time to refresh all temporary tables
        self.curs.close()
        self.conn.close()
        self.conn = sqlite3.connect(self.dbp)
        self.curs = self.conn.cursor()

        # If `removeOldVer` is True and an old database exists, delete it
        if removeOldVer and os.path.isfile(dstPath):
            os.remove(dstPath)
        topoOrder = self.getTopology()
        if len(topoOrder) == 0:
            return False

        allRootKeys = self.getAllRootKeys()

        # Create a series of temporary tables, these tables are all empty
        for tbn in topoOrder[::-1]:
            cmd = f"""
            CREATE TEMPORARY TABLE [Sampled{tbn}] AS SELECT * FROM [{tbn}] WHERE 1=0;
            """
            self.curs.execute(cmd)

        # Sample the tables to ensure they satisfy foreign key relationships
        for tbn in topoOrder[::-1]:
            if len(allRootKeys[tbn]) == 0:
                # No other tables reference `tbn` via foreign keys
                columnNames = self.getAllColumnNames(tbn)
                columnNames = [f'[{item}]' for item in columnNames] # Wrap every column name in [ ] to guard against spaces

                cmd = f"""
                INSERT INTO [Sampled{tbn}]
                  WITH [Ordered{tbn}] AS (
                    SELECT *, ROW_NUMBER() OVER (ORDER BY ROWID) AS row_num
                    FROM [{tbn}]
                  )
                  SELECT {', '.join(columnNames)}
                  FROM [Ordered{tbn}]
                  WHERE row_num IN (
                    SELECT row_num
                    FROM [Ordered{tbn}]
                    ORDER BY RANDOM()
                    LIMIT {sampleNum}
                  )
                  ORDER BY row_num;
                """ # This creates a temporary table that is valid throughout this connection, creating a VIEW would be permanently committed, using WITH is only valid for the current query, note the distinction
                self.curs.execute(cmd)
            else:
                # Maintain state to handle composite foreign keys, although the statement syntax may still be problematic
                artIdx = 0
                whereList = [allRootKeys[tbn][artIdx]]
                artIdx += 1
                while artIdx < len(allRootKeys[tbn]):
                    if whereList[-1]['linkedTable'] == allRootKeys[tbn][artIdx]:
                        whereList.append(allRootKeys[tbn][artIdx])
                    else:
                        if len(whereList) > 1:
                            print('There are more than 2 columns foreign key among 2 tables.')
                        whereStmt = ' AND '.join([f"[{tbn}].[{item['rootColumn']}] in (SELECT [Sampled{item['linkedTable']}].[{item['linkedColumn']}] FROM [Sampled{item['linkedTable']}])" for item in whereList])
                        cmd = f"""
                            INSERT INTO [Sampled{tbn}]
                              SELECT *
                              FROM [{tbn}]
                              WHERE {whereStmt};
                            """
                        self.curs.execute(cmd)
                        whereList = [allRootKeys[tbn][artIdx]]
                    artIdx += 1
                # Finally, don’t forget to flush any remaining items in `whereList`
                if len(whereList) > 1:
                    print('There are more than 2 columns foreign key among 2 tables.')
                whereStmt = ' AND '.join([f"[{tbn}].[{item['rootColumn']}] in (SELECT [Sampled{item['linkedTable']}].[{item['linkedColumn']}] FROM [Sampled{item['linkedTable']}])" for item in whereList])
                cmd = f"""
                  INSERT INTO [Sampled{tbn}]
                    SELECT *
                    FROM [{tbn}]
                    WHERE {whereStmt};
                  """
                self.curs.execute(cmd)

        # Save the results into another database
        zeroRow = False
        newConn = sqlite3.connect(dstPath)
        newCurs = newConn.cursor()
        for tbn in topoOrder:
            self.curs.execute(f'SELECT sql FROM sqlite_master WHERE type="table" AND name="{tbn}";')
            createSQL = self.curs.fetchall()[0][0]
            newCurs.execute(createSQL)
            self.curs.execute(f'SELECT * FROM [Sampled{tbn}];')
            rows = self.curs.fetchall()
            if len(rows) > 0:
                qCount = len(rows[0])
                qStr = ', '.join(['?' for _ in range(qCount)])
                newCurs.executemany(f'INSERT OR IGNORE INTO [{tbn}] VALUES ({qStr})', rows)
            else:
                zeroRow = True
                print(f'Sampled {self.dbn} exists 0 row table {tbn}.')
        newConn.commit()
        newConn.close()

        if zeroRow and removeEmpty:
            os.remove(dstPath)

    def getMergedTable(self):
        """
        Get tables that no other tables reference via foreign keys; these tables will be joined with
        others to produce a final large table. Rows from these large tables will later be sampled and
        fed to an LLM to generate text descriptions for fact‑verification examples.
        """
        pass

    @staticmethod
    def foreignKeyCheck(dbp):
        conn = sqlite3.connect(dbp)
        cur = conn.cursor()
        err = False
        try:
            cur.execute("PRAGMA foreign_keys = ON;")
            cur.execute("PRAGMA foreign_key_check;")
        except:
            err = True
        res = cur.fetchall()
        cur.close()
        conn.close()
        if err:
            return False
        if not res:
            return True
        return False

if __name__ == '__main__':
    dbRoot = '/home/zipengqiu/TableDatasetGeneration/dataset/workflowDB/'
    dbNames = os.listdir(dbRoot)
#    dbNames = ['address']

    for dbn in tqdm(dbNames):
        dbp = os.path.join(dbRoot, dbn, f'{dbn}.sqlite')
        db = DB(dbp)
        try:
            db.sample(f'dataset/{dbn}.sqlite', 1024)
        except Exception as E:
            print(E)
