import sqlite3
import sys
import argparse
from collections import defaultdict, Counter

from rich.console import Console
from rich.table import Table as RichTable
from rich.columns import Columns
from rich.panel import Panel
from rich import box


def get_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    return [row[0] for row in cur.fetchall()]


def get_qtypes(conn, table):
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT qtype FROM [{table}];")
    return [row[0] for row in cur.fetchall() if row[0] is not None]


def get_models(conn, tables):
    models = set()
    for table in tables:
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT DISTINCT model FROM [{table}];")
            models.update(row[0] for row in cur.fetchall() if row[0] is not None)
        except Exception:
            continue
    return sorted(models)


def fetch_stats(conn, table, model=None, qtype=None, allow_invalid=False, scale="8k"):
    cur = conn.cursor()
    
    # Check if validQuestion column exists
    cur.execute(f"PRAGMA table_info({table})")
    columns = [col[1] for col in cur.fetchall()]
    has_valid_question = 'validQuestion' in columns
    
    # Build query to filter out invalid questions (unless allow_invalid is True)
    query = f"SELECT correct, message FROM [{table}] WHERE 1=1"
    params = []
    
    # Only include rows where validQuestion=1 or validQuestion column doesn't exist
    # (unless allow_invalid flag is set)
    if has_valid_question and not allow_invalid:
        query += " AND (validQuestion = 1 OR validQuestion IS NULL)"
    
    # Filter by scale
    query += " AND scale=?"
    params.append(scale)
    
    if model:
        query += " AND model=?"
        params.append(model)
    if qtype:
        query += " AND qtype=?"
        params.append(qtype)
    
    cur.execute(query, params)
    rows = cur.fetchall()
    total = len(rows)
    correct = sum(1 for c, m in rows if c == 1)
    # Count number of times 'error' (case-insensitive) occurs in the message field
    errors = sum(1 for c, m in rows if m and 'error' in str(m).lower())
    incorrect = total - correct
    acc = correct / total if total > 0 else 0.0
    # Create scoreList: 1 for correct, 0 for incorrect
    scoreList = [1 if c == 1 else 0 for c, m in rows]
    return dict(total=total, correct=correct, incorrect=incorrect, errors=errors, accuracy=acc, scoreList=scoreList)


def accuracy_style(acc):
    if acc >= 0.9:
        return "bold green"
    elif acc >= 0.8:
        return "green"
    elif acc >= 0.5:
        return "yellow"
    elif acc >= 0.2:
        return "red"
    else:
        return "bold red"


def make_table(title, rows):
    table = RichTable(title=title, box=box.SQUARE)
    table.add_column("Label", justify="left")
    table.add_column("Total", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Incorrect", justify="right")
    table.add_column("'Errors'", justify="right")
    table.add_column("Accuracy", justify="right")
    for label, stats in rows:
        acc = stats['accuracy']
        style = accuracy_style(acc)
        label_style = style if (acc >= 0.9 or acc < 0.2) else None
        table.add_row(
            f"[{label_style}]{label}[/]" if label_style else label,
            str(stats['total']),
            str(stats['correct']),
            str(stats['incorrect']),
            str(stats['errors']),
            f"[{style}]{stats['accuracy']*100:.2f}%[/]"
        )
    return table


def make_comparison_table(label_order, stats1, stats2):
    table = RichTable(title="Comparison", box=box.SQUARE)
    table.add_column("Delta", justify="center")
    table.add_column("[green]A[/green][red]B[/red]", justify="center")
    table.add_column("[red]A[/red][green]B[/green]", justify="center")
    table.add_column("[red]A[/red][red]B[/red]", justify="center")
    table.add_column("[green]A[/green][green]B[/green]", justify="center")
    for label in label_order:
        correct1 = stats1.get(label, dict(correct=0)).get('correct', 0)
        correct2 = stats2.get(label, dict(correct=0)).get('correct', 0)
        scoreList1 = stats1.get(label, dict(scoreList=[])).get('scoreList', [])
        scoreList2 = stats2.get(label, dict(scoreList=[])).get('scoreList', [])
        
        delta = correct2 - correct1
        
        # Calculate the 4 comparison categories using actual score lists
        a_right_b_wrong = 0
        a_wrong_b_right = 0
        both_wrong = 0
        both_right = 0
        
        # Compare each question position (assuming same order)
        min_length = min(len(scoreList1), len(scoreList2))
        for i in range(min_length):
            a_score = scoreList1[i]
            b_score = scoreList2[i]
            
            if a_score == 1 and b_score == 0:
                a_right_b_wrong += 1
            elif a_score == 0 and b_score == 1:
                a_wrong_b_right += 1
            elif a_score == 0 and b_score == 0:
                both_wrong += 1
            elif a_score == 1 and b_score == 1:
                both_right += 1
        
        # Calculate improvement percentage
        total_compared = len(scoreList1) if len(scoreList1) == len(scoreList2) else min(len(scoreList1), len(scoreList2))
        improvement_pct = abs(delta / total_compared * 100) if total_compared > 0 else 0.0
        
        if delta > 0:
            delta_str = f"[green]+{delta} ({improvement_pct:.1f}%)[/green]"
        elif delta < 0:
            delta_str = f"[red]{delta} ({improvement_pct:.1f}%)[/red]"
        else:
            delta_str = f"0 ({improvement_pct:.1f}%)"
        
        # Color the numbers based on conditions
        a_right_b_wrong_str = f"[red]{a_right_b_wrong}[/red]" if a_right_b_wrong > 0 else str(a_right_b_wrong)
        a_wrong_b_right_str = f"[green]{a_wrong_b_right}[/green]" if a_wrong_b_right > 0 else str(a_wrong_b_right)
        
        table.add_row(
            delta_str,
            a_right_b_wrong_str,
            a_wrong_b_right_str,
            str(both_wrong),
            str(both_right)
        )
    return table


def analyze_sqlite(path, allow_invalid=False, scale="8k"):
    conn = sqlite3.connect(path)
    tables = get_tables(conn)
    models = get_models(conn, tables)
    if not models:
        raise RuntimeError(f"No models found in {path}")
    model = models[0] if len(models) == 1 else models[0]  # Always pick the first for now
    # Get qtype order from first table
    first_table_qtypes = get_qtypes(conn, tables[0]) if tables else []
    all_qtypes = list(first_table_qtypes)
    global_stats = Counter()
    global_qtype_stats = defaultdict(Counter)
    per_table_rows = {}
    for table in tables:
        table_qtypes = get_qtypes(conn, table)
        for q in table_qtypes:
            if q not in all_qtypes:
                all_qtypes.append(q)
        table_rows = []
        for qtype in table_qtypes:
            stats = fetch_stats(conn, table, model, qtype, allow_invalid, scale)
            table_rows.append((str(qtype), stats))
            for k in ["total", "correct", "incorrect", "errors"]:
                global_qtype_stats[qtype][k] += stats[k]
                global_stats[k] += stats[k]
        # Table summary row
        summary_stats = fetch_stats(conn, table, model, allow_invalid=allow_invalid, scale=scale)
        table_rows.append(("ALL", summary_stats))
        per_table_rows[table] = table_rows
    # Overall per qtype (across all tables) in the same order as first table
    overall_rows = []
    global_scoreLists = defaultdict(list)
    for qtype in all_qtypes:
        stats = global_qtype_stats[qtype]
        total = stats["total"]
        correct = stats["correct"]
        incorrect = stats["incorrect"]
        errors = stats["errors"]
        acc = correct / total if total > 0 else 0.0
        
        # Collect scoreLists for aggregation
        for table in tables:
            try:
                qtype_stats = fetch_stats(conn, table, model, qtype, allow_invalid, scale)
                if 'scoreList' in qtype_stats:
                    global_scoreLists[qtype].extend(qtype_stats['scoreList'])
            except Exception:
                continue
        
        overall_rows.append((str(qtype), dict(total=total, correct=correct, incorrect=incorrect, errors=errors, accuracy=acc, scoreList=global_scoreLists[qtype])))
    
    # Global summary row - aggregate all scoreLists
    total = global_stats["total"]
    correct = global_stats["correct"]
    incorrect = global_stats["incorrect"]
    errors = global_stats["errors"]
    acc = correct / total if total > 0 else 0.0
    
    all_scoreList = []
    for qtype in all_qtypes:
        all_scoreList.extend(global_scoreLists[qtype])
    
    overall_rows.append(("ALL", dict(total=total, correct=correct, incorrect=incorrect, errors=errors, accuracy=acc, scoreList=all_scoreList)))
    return {
        "model": model,
        "tables": tables,
        "per_table_rows": per_table_rows,
        "overall_rows": overall_rows,
        "qtype_order": all_qtypes + ["ALL"]
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare results from two SQLite files")
    parser.add_argument("path1", help="Path to first SQLite file")
    parser.add_argument("path2", help="Path to second SQLite file")
    parser.add_argument("--allow-invalid", action="store_true", 
                       help="Include invalid questions in analysis (default: filter out invalid questions)")
    parser.add_argument("--scalea", default="8k", 
                       help="Scale of data to analyze for first file (default: 8k)")
    parser.add_argument("--scaleb", default="8k", 
                       help="Scale of data to analyze for second file (default: 8k)")
    
    args = parser.parse_args()
    
    console = Console()
    try:
        res1 = analyze_sqlite(args.path1, allow_invalid=args.allow_invalid, scale=args.scalea)
        res2 = analyze_sqlite(args.path2, allow_invalid=args.allow_invalid, scale=args.scaleb)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    # Use the union of all tables, but keep order from first
    all_tables = res1["tables"]
    # Panel titles
    left_title = f"{res1['model']} (scale: {args.scalea})"
    right_title = f"{res2['model']} (scale: {args.scaleb})"
    # For each table
    for table in all_tables:
        rows1 = res1["per_table_rows"].get(table, [])
        rows2 = res2["per_table_rows"].get(table, [])
        # Align rows by label order
        label_order = [label for label, _ in rows1]
        stats1 = {label: stats for label, stats in rows1}
        stats2 = {label: stats for label, stats in rows2}
        aligned_rows1 = [(label, stats1.get(label, dict(total=0, correct=0, incorrect=0, errors=0, accuracy=0.0))) for label in label_order]
        aligned_rows2 = [(label, stats2.get(label, dict(total=0, correct=0, incorrect=0, errors=0, accuracy=0.0))) for label in label_order]
        table1 = make_table(left_title, aligned_rows1)
        table2 = make_table(right_title, aligned_rows2)
        delta_table = make_comparison_table(label_order, stats1, stats2)
        panel = Panel.fit(
            Columns([table1, table2, delta_table], padding=(1, 4)),
            title=f"Table: {table}",
            border_style="cyan",
            title_align="center",
            padding=(1, 2),
        )
        console.print(panel)
    # Overall summary
    label_order = [label for label, _ in res1["overall_rows"]]
    stats1 = {label: stats for label, stats in res1["overall_rows"]}
    stats2 = {label: stats for label, stats in res2["overall_rows"]}
    aligned_rows1 = [(label, stats1.get(label, dict(total=0, correct=0, incorrect=0, errors=0, accuracy=0.0))) for label in label_order]
    aligned_rows2 = [(label, stats2.get(label, dict(total=0, correct=0, incorrect=0, errors=0, accuracy=0.0))) for label in label_order]
    delta_table = make_comparison_table(label_order, stats1, stats2)
    table1 = make_table(left_title, aligned_rows1)
    table2 = make_table(right_title, aligned_rows2)
    panel = Panel.fit(
        Columns([table1, table2, delta_table], padding=(1, 4)),
        title="Overall per QType and Global",
        border_style="magenta",
        title_align="center",
        padding=(1, 2),
    )
    console.print(panel)

if __name__ == "__main__":
    main() 