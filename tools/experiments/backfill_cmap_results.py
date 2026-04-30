#!/usr/bin/env python3
"""
Re-parse existing cmap_parallel log files and rewrite cmap_results.json
with full per_class_per_threshold_cAP data.

Usage:
    python3 tools/experiments/backfill_cmap_results.py \
        --log LOG_FILE \
        --suite-root SUITE_ROOT \
        [--dry-run]
"""
import ast
import re
import json
import sys
import argparse
from pathlib import Path


COND_START = re.compile(r"\[cmap-parallel\] === Starting (\S+) ===")
COND_FINISH = re.compile(r"\[cmap-parallel\] === Finished (\S+) ===")
CONS_FRAMES_RE = re.compile(r"--cons_frames\s+(\d+)")


def parse_condition_section(section_text: str, condition: str, cons_frames: int) -> dict:
    per_cls_thr: dict[str, dict[str, float]] = {}
    for line in section_text.splitlines():
        line = line.strip()
        if "AP@" in line and line.startswith("{"):
            try:
                d = ast.literal_eval(line)
                for cls, thr_dict in d.items():
                    if cls not in per_cls_thr:
                        per_cls_thr[cls] = {}
                    for thr_key, val in thr_dict.items():
                        thr = thr_key.split("@")[1]
                        per_cls_thr[cls][thr] = val
            except Exception:
                pass

    # per_threshold_cAP: mean across classes at each threshold
    per_thr: dict[str, list[float]] = {}
    for cls, thr_vals in per_cls_thr.items():
        for thr, val in thr_vals.items():
            per_thr.setdefault(thr, []).append(val)
    per_thr_mean = {thr: sum(v) / len(v) for thr, v in per_thr.items()}

    # category_cAP
    cat_cap: dict[str, float] = {}
    cm = re.search(r"Category mean AP\s*(\{[^}]+\})", section_text)
    if cm:
        try:
            cat_cap = ast.literal_eval(cm.group(1))
        except Exception:
            pass

    # mean_cMAP
    mm = re.search(r"^mean AP\s+([\d.]+)", section_text, re.MULTILINE)
    mean_cmap = float(mm.group(1)) if mm else None

    return {
        "condition": condition,
        "cons_frames": cons_frames,
        "per_class_per_threshold_cAP": per_cls_thr,
        "per_threshold_cAP": per_thr_mean,
        "category_cAP": cat_cap,
        "mean_cMAP": mean_cmap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to cmap_seq_*.out log file")
    parser.add_argument("--suite-root", required=True, help="Suite root dir containing condition subdirs")
    parser.add_argument("--dry-run", action="store_true", help="Print parsed results without writing")
    args = parser.parse_args()

    log_path = Path(args.log)
    suite_root = Path(args.suite_root)

    if not log_path.exists():
        print(f"ERROR: log not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    if not suite_root.exists():
        print(f"ERROR: suite-root not found: {suite_root}", file=sys.stderr)
        sys.exit(1)

    text = log_path.read_text()

    # Detect cons_frames from log
    cfm = CONS_FRAMES_RE.search(text)
    cons_frames = int(cfm.group(1)) if cfm else 5

    # Split into condition sections
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []  # [(cond, section_text)]
    current_cond = None
    current_lines: list[str] = []

    for line in lines:
        sm = COND_START.search(line)
        fm = COND_FINISH.search(line)
        if sm:
            current_cond = sm.group(1)
            current_lines = []
        elif fm and current_cond:
            sections.append((current_cond, "\n".join(current_lines)))
            current_cond = None
            current_lines = []
        elif current_cond:
            current_lines.append(line)

    if not sections:
        print("WARNING: No condition sections found in log. Treating entire log as one section.")
        # Fallback: try to get condition from 'Wrote ... cmap_results.json' line
        wrote_m = re.search(r"\] Wrote (.+)/cmap_results\.json", text)
        if wrote_m:
            cond_dir = Path(wrote_m.group(1))
            cond = cond_dir.name
            sections = [(cond, text)]
        else:
            print("ERROR: Cannot determine conditions from log.", file=sys.stderr)
            sys.exit(1)

    print(f"Found {len(sections)} condition(s) in log: {[c for c,_ in sections]}")

    for cond, section_text in sections:
        out = parse_condition_section(section_text, cond, cons_frames)
        if out["mean_cMAP"] is None:
            print(f"  WARNING: could not parse mean_cMAP for {cond}, skipping")
            continue

        cond_dir = suite_root / cond
        json_path = cond_dir / "cmap_results.json"

        if args.dry_run:
            print(f"\n[DRY RUN] {cond}:")
            print(json.dumps(out, indent=2))
        else:
            if not cond_dir.exists():
                print(f"  WARNING: condition dir not found: {cond_dir}, skipping")
                continue
            json_path.write_text(json.dumps(out, indent=2) + "\n")
            print(f"  Wrote {json_path}  mean_cMAP={out['mean_cMAP']:.6f}  "
                  f"ped={out['category_cAP'].get('ped_crossing','?'):.4f}  "
                  f"div={out['category_cAP'].get('divider','?'):.4f}  "
                  f"bnd={out['category_cAP'].get('boundary','?'):.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
