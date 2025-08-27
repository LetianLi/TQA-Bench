#!/usr/bin/env python3
"""
Script to reanalyze results by reextracting answers from message columns
and comparing them to existing pred columns in SQLite files.
"""

import sqlite3
import sys
import os
import glob
import argparse
from pathlib import Path

from symDataloader.utils import extractAnswer, testValid

def get_tables(conn):
    """Get all table names from SQLite database."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    return [row[0] for row in cur.fetchall()]


def reextractAnswer(sqlite_paths, dry_run=True):
    """
    Reextract answers from message columns and compare to pred columns.
    
    Args:
        sqlite_paths: List of paths to SQLite files
        dry_run: If True, only report changes without persisting them
    """
    total_files = len(sqlite_paths)
    total_changes = 0
    
    print(f"Processing {total_files} SQLite file(s)...")
    print(f"Dry run mode: {'ON' if dry_run else 'OFF'}")
    print("-" * 50)
    
    for file_path in sqlite_paths:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        
        try:
            conn = sqlite3.connect(file_path)
            tables = get_tables(conn)
            
            if not tables:
                print(f"  No tables found in {os.path.basename(file_path)}")
                continue
                
            file_changes = 0
            
            for table in tables:
                print(f"  Table: {table}")
                table_changes = 0
                
                # Get all rows with message, pred, gt, and correct columns
                cur = conn.cursor()
                try:
                    cur.execute(f"SELECT rowid, message, pred, gt, correct FROM [{table}] WHERE message IS NOT NULL")
                    rows = cur.fetchall()
                except sqlite3.OperationalError as e:
                    print(f"    Error reading table {table}: {e}")
                    continue
                
                for rowid, message, current_pred, gt, current_correct in rows:
                    if message is None:
                        continue
                        
                    # Extract new answer from message
                    new_pred = extractAnswer(message)
                    
                    # Calculate new correct value
                    new_correct = 1 if new_pred == gt else 0
                    
                    # Compare with current prediction or correctness
                    if new_pred != current_pred or new_correct != current_correct:
                        if new_pred != current_pred:
                            print(f"    Row {rowid}: pred changed from '{current_pred}' to '{new_pred}'")
                        if new_correct != current_correct:
                            print(f"    Row {rowid}: correct changed from {current_correct} to {new_correct}")
                        table_changes += 1
                        
                        # Update the database if not in dry run mode
                        if not dry_run:
                            try:
                                cur.execute(f"UPDATE [{table}] SET pred = ?, correct = ? WHERE rowid = ?", (new_pred, new_correct, rowid))
                            except sqlite3.OperationalError as e:
                                print(f"      Error updating row {rowid}: {e}")
                
                if table_changes > 0:
                    print(f"    Total changes in table '{table}': {table_changes}")
                    file_changes += table_changes
                else:
                    print(f"    No changes in table '{table}'")
                
                # Commit changes for this table if not in dry run mode
                if not dry_run and table_changes > 0:
                    conn.commit()
            
            if file_changes > 0:
                print(f"  Total changes in file: {file_changes}")
                total_changes += file_changes
            else:
                print(f"  No changes in file")
                
            conn.close()
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(file_path)}: {e}")
            continue
    
    print("\n" + "=" * 50)
    print(f"Summary:")
    print(f"  Files processed: {total_files}")
    print(f"  Total changes: {total_changes}")
    if dry_run:
        print(f"  No changes were persisted (dry run mode)")
    else:
        print(f"  All changes have been persisted to the database(s)")

def validateQuestion(sqlite_paths, dry_run=True):
    """
    Validate questions by checking against the dataset and adding validQuestion column.
    
    Args:
        sqlite_paths: List of paths to SQLite files
        dry_run: If True, only report changes without persisting them
    """
    total_files = len(sqlite_paths)
    total_invalid = 0
    
    print(f"Validating questions in {total_files} SQLite file(s)...")
    print(f"Dry run mode: {'ON' if dry_run else 'OFF'}")
    print("-" * 50)
    
    # Connect to the dataset once
    try:
        dataset_conn = sqlite3.connect('symDataset/tasks/TableQA/dataset.sqlite')
        print("Connected to dataset.sqlite")
    except Exception as e:
        print(f"Error connecting to dataset: {e}")
        return
    
    for file_path in sqlite_paths:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        
        try:
            conn = sqlite3.connect(file_path)
            tables = get_tables(conn)
            
            if not tables:
                print(f"  No tables found in {os.path.basename(file_path)}")
                continue
                
            file_invalid = 0
            
            for table in tables:
                print(f"  Table: {table}")
                table_invalid = 0
                
                # Check if validQuestion column exists, create if not
                cur = conn.cursor()
                try:
                    cur.execute(f"PRAGMA table_info({table})")
                    columns = [col[1] for col in cur.fetchall()]
                    
                    if 'validQuestion' not in columns:
                        if not dry_run:
                            cur.execute(f"ALTER TABLE [{table}] ADD COLUMN validQuestion INTEGER DEFAULT 0")
                            conn.commit()
                            print(f"    Added validQuestion column to table '{table}'")
                        else:
                            print(f"    Would add validQuestion column to table '{table}' (dry run)")
                
                    # Get all rows with the required columns
                    cur.execute(f"SELECT rowid, scale, dbIdx, sampleIdx, questionIdx FROM [{table}]")
                    rows = cur.fetchall()
                    
                    for rowid, scale, dbIdx, sampleIdx, questionIdx in rows:
                        if None in (scale, dbIdx, sampleIdx, questionIdx):
                            continue
                            
                        # Query the dataset to get the question details
                        dataset_cur = dataset_conn.cursor()
                        try:
                            dataset_cur.execute(f"""
                                SELECT rightIdx, A, B, C, D 
                                FROM [{table}] 
                                WHERE scale=? AND dbIdx=? AND sampleIdx=? AND questionIdx=?
                            """, (scale, dbIdx, sampleIdx, questionIdx))
                            
                            dataset_row = dataset_cur.fetchone()
                            if dataset_row:
                                rightIdx, choice_a, choice_b, choice_c, choice_d = dataset_row
                                
                                # Check if the question is valid using the private function
                                is_valid, reason = testValid(rightIdx, choice_a, choice_b, choice_c, choice_d)
                                
                                # Check current validQuestion value (if column exists)
                                current_valid = None  # Use None to indicate column doesn't exist
                                if 'validQuestion' in columns:
                                    cur.execute(f"SELECT validQuestion FROM [{table}] WHERE rowid = ?", (rowid,))
                                    current_valid_result = cur.fetchone()
                                    current_valid = current_valid_result[0] if current_valid_result else 0
                                
                                # Update if different or if column doesn't exist yet
                                new_valid = 1 if is_valid else 0
                                should_update = (current_valid is None) or (new_valid != current_valid)
                                
                                # Always count invalid questions
                                if new_valid == 0:
                                    print(f"    Row {rowid}: Invalid question detected (scale={scale}, dbIdx={dbIdx}, sampleIdx={sampleIdx}, questionIdx={questionIdx}) - {reason}")
                                    table_invalid += 1
                                
                                # Update if different or if column doesn't exist yet
                                if should_update:
                                    # Update the database if not in dry run mode
                                    if not dry_run:
                                        try:
                                            cur.execute(f"UPDATE [{table}] SET validQuestion = ? WHERE rowid = ?", (new_valid, rowid))
                                        except sqlite3.OperationalError as e:
                                            print(f"      Error updating row {rowid}: {e}")
                                
                        except Exception as e:
                            print(f"      Error querying dataset for row {rowid}: {e}")
                            continue
                
                except sqlite3.OperationalError as e:
                    print(f"    Error processing table {table}: {e}")
                    continue
                
                if table_invalid > 0:
                    print(f"    Invalid questions in table '{table}': {table_invalid}")
                    file_invalid += table_invalid
                else:
                    print(f"    All questions in table '{table}' are valid")
                
                # Commit changes for this table if not in dry run mode
                if not dry_run:
                    conn.commit()
            
            if file_invalid > 0:
                print(f"  Total invalid questions in file: {file_invalid}")
                total_invalid += file_invalid
            else:
                print(f"  All questions in file are valid")
                
            conn.close()
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(file_path)}: {e}")
            continue
    
    # Close dataset connection
    dataset_conn.close()
    
    print("\n" + "=" * 50)
    print(f"Summary:")
    print(f"  Files processed: {total_files}")
    print(f"  Total invalid questions: {total_invalid}")
    if dry_run:
        print(f"  No changes were persisted (dry run mode)")
    else:
        print(f"  All changes have been persisted to the database(s)")


def inspectQuestionChoice(sqlite_paths, skip_invalid=True):
    """
    Inspect incorrect answers and show answer choices, correct answer, and model prediction.
    
    Args:
        sqlite_paths: List of paths to SQLite files
    """
    total_files = len(sqlite_paths)
    total_incorrect = 0
    
    print(f"Inspecting incorrect answers in {total_files} SQLite file(s)...")
    print("-" * 50)
    
    # Connect to the dataset once
    try:
        dataset_conn = sqlite3.connect('symDataset/tasks/TableQA/dataset.sqlite')
        print("Connected to dataset.sqlite")
    except Exception as e:
        print(f"Error connecting to dataset: {e}")
        return
    
    for file_path in sqlite_paths:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        
        try:
            conn = sqlite3.connect(file_path)
            tables = get_tables(conn)
            
            if not tables:
                print(f"  No tables found in {os.path.basename(file_path)}")
                continue
                
            file_incorrect = 0
            
            for table in tables:
                print(f"  Table: {table}")
                table_incorrect = 0
                
                # Get all incorrect answers (correct=0) and skip invalid questions
                cur = conn.cursor()
                try:
                    # Get rows with incorrect answers, excluding invalid questions
                    cur.execute(f"""
                        SELECT rowid, scale, dbIdx, sampleIdx, questionIdx, pred, gt 
                        FROM [{table}] 
                        WHERE correct = 0
                        {'AND (validQuestion = 1 OR validQuestion IS NULL)' if skip_invalid else ''}
                    """)
                    rows = cur.fetchall()
                    
                    for rowid, scale, dbIdx, sampleIdx, questionIdx, pred, gt in rows:
                        if None in (scale, dbIdx, sampleIdx, questionIdx):
                            continue
                            
                        # Query the dataset to get the question details
                        dataset_cur = dataset_conn.cursor()
                        try:
                            dataset_cur.execute(f"""
                                SELECT rightIdx, A, B, C, D 
                                FROM [{table}] 
                                WHERE scale=? AND dbIdx=? AND sampleIdx=? AND questionIdx=?
                            """, (scale, dbIdx, sampleIdx, questionIdx))
                            
                            dataset_row = dataset_cur.fetchone()
                            if dataset_row:
                                rightIdx, choice_a, choice_b, choice_c, choice_d = dataset_row
                                
                                # Check if the question is valid (skip invalid questions)
                                is_valid, reason = testValid(rightIdx, choice_a, choice_b, choice_c, choice_d)
                                if not is_valid:
                                    continue
                                
                                # Determine which choice is correct
                                choices = [choice_a, choice_b, choice_c, choice_d]
                                if rightIdx is not None and 0 <= rightIdx < len(choices):
                                    correct_choice = choices[rightIdx]
                                    correct_letter = ['A', 'B', 'C', 'D'][rightIdx]
                                else:
                                    correct_choice = "Unknown"
                                    correct_letter = "?"
                                
                                # Print the analysis
                                print(f"    Row {rowid} (scale={scale}, dbIdx={dbIdx}, sampleIdx={sampleIdx}, questionIdx={questionIdx}):")
                                print(f"      Choices: A={choice_a}, B={choice_b}, C={choice_c}, D={choice_d}")
                                print(f"      Correct: {correct_letter} ({correct_choice})")
                                print(f"      Model answered: {pred}")
                                print(f"      Ground truth: {gt}")
                                print()
                                
                                table_incorrect += 1
                                
                        except Exception as e:
                            print(f"      Error querying dataset for row {rowid}: {e}")
                            continue
                
                except sqlite3.OperationalError as e:
                    print(f"    Error processing table {table}: {e}")
                    continue
                
                if table_incorrect > 0:
                    print(f"    Total incorrect answers in table '{table}': {table_incorrect}")
                    file_incorrect += table_incorrect
                else:
                    print(f"    No incorrect answers found in table '{table}'")
            
            if file_incorrect > 0:
                print(f"  Total incorrect answers in file: {file_incorrect}")
                total_incorrect += file_incorrect
            else:
                print(f"  No incorrect answers found in file")
                
            conn.close()
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(file_path)}: {e}")
            continue
    
    # Close dataset connection
    dataset_conn.close()
    
    print("\n" + "=" * 50)
    print(f"Summary:")
    print(f"  Files processed: {total_files}")
    print(f"  Total incorrect answers analyzed: {total_incorrect}")


def main():
    parser = argparse.ArgumentParser(description="Reanalyze results by reextracting answers from SQLite files")
    parser.add_argument("command", help="Command to execute (e.g., 'reextractAnswer', 'validateQuestion', 'inspectQuestionChoice')")
    parser.add_argument("files", nargs="*", help="SQLite file names or patterns (supports wildcards like '5_mini*') or --all flag")
    parser.add_argument("--all", action="store_true", help="Process all SQLite files in symDataset/results/TableQA")
    parser.add_argument("--dry", action="store_true", help="Dry run mode - report changes without persisting them")
    
    args = parser.parse_args()
    
    if args.command not in ["reextractAnswer", "validateQuestion", "inspectQuestionChoice"]:
        print(f"Unknown command: {args.command}")
        print("Available commands: reextractAnswer, validateQuestion, inspectQuestionChoice")
        sys.exit(1)
    
    # Determine which files to process
    sqlite_files = []
    
    if args.all:
        # Get all SQLite files from the directory
        results_dir = Path("symDataset/results/TableQA")
        if not results_dir.exists():
            print(f"Error: Directory {results_dir} does not exist")
            sys.exit(1)
        
        sqlite_files = list(results_dir.glob("*.sqlite"))
        if not sqlite_files:
            print(f"No SQLite files found in {results_dir}")
            sys.exit(1)
    elif args.files:
        # Process specific files (support wildcards)
        results_dir = Path("symDataset/results/TableQA")
        for pattern in args.files:
            # Use glob to expand wildcards
            matched_files = list(results_dir.glob(pattern))
            if not matched_files:
                print(f"Warning: No files found matching pattern '{pattern}' in {results_dir}")
                continue
            
            # Filter to only include .sqlite files
            sqlite_matches = [f for f in matched_files if f.suffix == '.sqlite']
            if not sqlite_matches:
                print(f"Warning: No .sqlite files found matching pattern '{pattern}' in {results_dir}")
                continue
                
            sqlite_files.extend(sqlite_matches)
            print(f"Pattern '{pattern}' matched {len(sqlite_matches)} SQLite files")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file_path in sqlite_files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)
        
        sqlite_files = unique_files
    else:
        print("Error: Must specify either --all flag or specific file names")
        sys.exit(1)
    
    if not sqlite_files:
        print("No valid SQLite files to process")
        sys.exit(1)
    
    # Execute the command
    if args.command == "reextractAnswer":
        reextractAnswer(sqlite_files, dry_run=args.dry)
    elif args.command == "validateQuestion":
        validateQuestion(sqlite_files, dry_run=args.dry)
    elif args.command == "inspectQuestionChoice":
        inspectQuestionChoice(sqlite_files, True)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
