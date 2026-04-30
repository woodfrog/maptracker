"""Probe B0 corruption suite results to understand data structure and compute mAP."""
import json
import os
import sys
from pathlib import Path

SUITE_ROOT = Path("work_dirs/experiments/b0/corruption_suite/latest_onset0_trainmatched")

# ---- Probe submission_vector structure ----
cond = "c_full_offset1_onset0"
sv = json.load(open(SUITE_ROOT / cond / "submission_vector.json"))
results = sv.get("results", {})
first_key = list(results.keys())[0]
first_val = results[first_key]

print("=== submission_vector structure ===")
print(f"  top keys: {list(sv.keys())}")
print(f"  # results (samples): {len(results)}")
print(f"  first sample key: {first_key}")
print(f"  first sample type: {type(first_val)}")
if isinstance(first_val, dict):
    print(f"  first sample dict keys: {list(first_val.keys())}")
    for k, v in first_val.items():
        if isinstance(v, list):
            print(f"    {k}: list len={len(v)}")
            if v:
                print(f"      first item keys: {list(v[0].keys()) if isinstance(v[0], dict) else type(v[0])}")
        else:
            print(f"    {k}: {v}")
elif isinstance(first_val, list):
    print(f"  first sample list len: {len(first_val)}")
    if first_val and isinstance(first_val[0], dict):
        print(f"  first item keys: {list(first_val[0].keys())}")

print()

# ---- Check alpha_stats ----
alpha = json.load(open(SUITE_ROOT / cond / "alpha_stats.json"))
print("=== alpha_stats structure ===")
def show(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            show(v, indent + 2)
        elif isinstance(v, list):
            print(" " * indent + f"{k}: [list len={len(v)}]")
        else:
            print(" " * indent + f"{k}: {v}")
show(alpha)

print()

# ---- Try to find mAP eval results (look in the condition workdir logs or eval dir) ----
print("=== All files in condition dir ===")
for p in sorted((SUITE_ROOT / cond).iterdir()):
    size = p.stat().st_size if p.is_file() else "-"
    print(f"  {p.name}  ({size} bytes)")

print()

# ---- Check if there's an eval results json in any subdir ----
print("=== Searching for eval/metric json files under suite root ===")
for p in SUITE_ROOT.rglob("*.json"):
    if p.name not in ("submission_vector.json", "contradiction_metrics.json", "alpha_stats.json", "condition_meta.json", "contradiction_suite_summary.json"):
        print(f"  {p.relative_to(SUITE_ROOT)}")
