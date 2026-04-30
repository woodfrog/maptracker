"""
Parse MapTracker training logs into CSVs.
Handles the eta field which contains a comma: "5 days, 11:50:59".
"""
import re
import csv
import os

LOG_DIR = os.path.dirname(os.path.abspath(__file__))

SECTIONS = [
    # (log_file, start_line, end_line_inclusive, output_csv)
    ("B1Train.log", 1477,   103496, "OldB1TrainingLogs.csv"),
    ("B1Train.log", 103497, None,   "FinalB1TrainingLogs.csv"),
    ("B2Train.log", 1477,   91829,  "OldB2TrainingLogs.csv"),
    ("B2Train.log", 91830,  None,   "FinalB2TrainingLogs.csv"),
]

# Regex to detect an Iter log line
ITER_RE = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - mmdet - INFO - Iter \[(\d+)/(\d+)\]\s+(.*)'
)


def parse_iter_line(line):
    m = ITER_RE.match(line.rstrip())
    if not m:
        return None
    timestamp, iter_num, max_iters, metrics_str = m.groups()

    # Normalise eta: "5 days, 11:50:59"  ->  "5 days 11:50:59"
    # so the comma-split later stays clean
    metrics_str = re.sub(r'(\d+ days?), (\d+:\d+:\d+)', r'\1 \2', metrics_str)

    row = {'timestamp': timestamp, 'iter': int(iter_num)}

    for kv in metrics_str.split(', '):
        kv = kv.strip()
        if ': ' not in kv:
            continue
        key, val = kv.split(': ', 1)
        key = key.strip()
        val = val.strip()
        try:
            row[key] = float(val)
        except ValueError:
            row[key] = val  # keep eta as a string

    return row


def process_section(log_filename, start_line, end_line, output_csv):
    log_path = os.path.join(LOG_DIR, log_filename)
    out_path  = os.path.join(LOG_DIR, output_csv)

    print(f"  Reading {log_filename} lines {start_line}–{end_line or 'EOF'} → {output_csv}")
    rows = []

    # Some log sections are prefixed with a GPU rank like "0: " from distributed training
    rank_prefix_re = re.compile(r'^\d+: ')

    with open(log_path, 'r', errors='replace') as fh:
        for lineno, line in enumerate(fh, 1):
            if lineno < start_line:
                continue
            if end_line and lineno > end_line:
                break
            # Strip optional rank prefix (e.g. "0: ")
            line = rank_prefix_re.sub('', line, count=1)
            if '- mmdet - INFO - Iter [' not in line:
                continue
            row = parse_iter_line(line)
            if row:
                rows.append(row)

    if not rows:
        print(f"    WARNING: no iteration lines found!")
        return

    # Collect all column names in first-seen order
    all_keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(out_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"    Wrote {len(rows):,} rows, {len(all_keys)} columns → {output_csv}")


if __name__ == '__main__':
    for args in SECTIONS:
        process_section(*args)
    print("Done.")
