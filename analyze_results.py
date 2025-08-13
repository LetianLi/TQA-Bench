import sqlite3
import sys
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


def fetch_stats(conn, table, model=None, qtype=None):
    cur = conn.cursor()
    query = f"SELECT correct, message FROM [{table}] WHERE 1=1"
    params = []
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
    return dict(total=total, correct=correct, incorrect=incorrect, errors=errors, accuracy=acc)


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


def make_delta_table(label_order, stats1, stats2):
    table = RichTable(title="Delta", box=box.SQUARE)
    table.add_column("Delta", justify="center")
    for label in label_order:
        correct1 = stats1.get(label, dict(correct=0)).get('correct', 0)
        correct2 = stats2.get(label, dict(correct=0)).get('correct', 0)
        delta = correct2 - correct1
        if delta > 0:
            style = "green"
            delta_str = f"+{delta}"
        elif delta < 0:
            style = "red"
            delta_str = f"{delta}"
        else:
            style = None
            delta_str = "+0"
        table.add_row(f"[{style}]{delta_str}[/]" if style else delta_str)
    return table


def analyze_sqlite(path):
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
            stats = fetch_stats(conn, table, model, qtype)
            table_rows.append((str(qtype), stats))
            for k in ["total", "correct", "incorrect", "errors"]:
                global_qtype_stats[qtype][k] += stats[k]
                global_stats[k] += stats[k]
        # Table summary row
        summary_stats = fetch_stats(conn, table, model)
        table_rows.append(("ALL", summary_stats))
        per_table_rows[table] = table_rows
    # Overall per qtype (across all tables) in the same order as first table
    overall_rows = []
    for qtype in all_qtypes:
        stats = global_qtype_stats[qtype]
        total = stats["total"]
        correct = stats["correct"]
        incorrect = stats["incorrect"]
        errors = stats["errors"]
        acc = correct / total if total > 0 else 0.0
        overall_rows.append((str(qtype), dict(total=total, correct=correct, incorrect=incorrect, errors=errors, accuracy=acc)))
    # Global summary row
    total = global_stats["total"]
    correct = global_stats["correct"]
    incorrect = global_stats["incorrect"]
    errors = global_stats["errors"]
    acc = correct / total if total > 0 else 0.0
    overall_rows.append(("ALL", dict(total=total, correct=correct, incorrect=incorrect, errors=errors, accuracy=acc)))
    return {
        "model": model,
        "tables": tables,
        "per_table_rows": per_table_rows,
        "overall_rows": overall_rows,
        "qtype_order": all_qtypes + ["ALL"]
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: uv run analyze_results.py path1 path2")
        sys.exit(1)
    path1, path2 = sys.argv[1], sys.argv[2]
    console = Console()
    try:
        res1 = analyze_sqlite(path1)
        res2 = analyze_sqlite(path2)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    # Use the union of all tables, but keep order from first
    all_tables = res1["tables"]
    # Panel titles
    left_title = f"{res1['model']}"
    right_title = f"{res2['model']}"
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
        delta_table = make_delta_table(label_order, stats1, stats2)
        table1 = make_table(left_title, aligned_rows1)
        table2 = make_table(right_title, aligned_rows2)
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
    delta_table = make_delta_table(label_order, stats1, stats2)
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