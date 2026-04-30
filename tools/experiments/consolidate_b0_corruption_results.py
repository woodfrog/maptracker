#!/usr/bin/env python3
"""
Consolidate B0 corruption suite results into a wide-format CSV and Markdown summary.

Parses per-condition eval logs (mAP_normal + per-category AP) together with
per-condition contradiction_metrics.json and alpha_stats.json.

Usage (from maptracker_v1 root):
    python tools/experiments/consolidate_b0_corruption_results.py [options]

Options:
    --suite-root DIR     Path to the B0 corruption suite directory
                         (default: work_dirs/experiments/b0/corruption_suite/latest_onset0_trainmatched)
    --log-dir DIR        Path to sbatch log dir containing *.out files
                         (default: auto-detected from most recent sbatch run)
    --out-dir DIR        Where to write outputs (default: CurrentB1B2Results/b0)

Outputs:
    corrupted_eval_results_b0.csv        – per model x condition x category (matches B1/B2 format)
    contradiction_results_b0.csv         – wide format, one row per condition (matches master format)
    contradiction_results_b0_summary.md  – Markdown tables
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

CONDITIONS = [
    "c_full_offset1_onset0",
    "c_full_offset2_onset0",
    "c_full_offset3_onset0",
    "c_tail_offset1_onset0",
    "c_tail_offset2_onset0",
    "c_tail_offset3_onset0",
]

CATEGORIES = ["ped_crossing", "divider", "boundary"]

WIDE_COLUMNS = (
    ["model", "condition_tag", "mode", "stale_offset", "onset"]
    + ["stale_persistence_time_mean", "cumulative_stale_FP_polyline_length", "alpha_separation_summary"]
    + ["alpha_mean_affected", "alpha_mean_clean_recent", "alpha_mean_preserved_recent"]
    + [f"{cat}_num_preds" for cat in CATEGORIES]
    + [f"{cat}_num_gts"   for cat in CATEGORIES]
    + [f"{cat}_AP_0.5"    for cat in CATEGORIES]
    + [f"{cat}_AP_1.0"    for cat in CATEGORIES]
    + [f"{cat}_AP_1.5"    for cat in CATEGORIES]
    + [f"{cat}_AP"        for cat in CATEGORIES]
    + ["mAP", "mean_mAP_across_conditions"]
    # cMAP columns (populated when cmap_results.json exists)
    + [f"{cat}_cAP" for cat in CATEGORIES]
    + ["mean_cMAP", "mean_cMAP_across_conditions"]
)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_eval_log(log_path: Path):
    """
    Parse a .out log file from run_contradiction_suite and extract:
      - condition_tag (from work_dir line)
      - per-category: num_preds, num_gts, AP@0.5, AP@1.0, AP@1.5, AP
      - mAP_normal
    Returns None if the log doesn't contain the expected output.
    """
    text = log_path.read_text(errors="replace")

    # Extract condition tag from the work_dir line
    cond_m = re.search(r"work_dir:\s+\S+/(\w+_offset\d+_onset\d+)", text)
    if not cond_m:
        return None
    condition_tag = cond_m.group(1)

    # Extract mAP_normal
    map_m = re.search(r"mAP_normal\s*=\s*([\d.]+)", text)
    if not map_m:
        return None
    mAP_normal = float(map_m.group(1))

    # Extract per-category table rows: "| category | num_preds | num_gts | AP@0.5 | AP@1.0 | AP@1.5 | AP |"
    # Pattern: | ped_crossing |   175341  |   6922  | 0.6556 | 0.8428 | 0.8976 | 0.7987 |
    row_pattern = re.compile(
        r"\|\s*(ped_crossing|divider|boundary)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
        r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    )

    cat_data = {}
    for m in row_pattern.finditer(text):
        cat = m.group(1)
        cat_data[cat] = {
            "num_preds": int(m.group(2)),
            "num_gts":   int(m.group(3)),
            "AP_0.5":    float(m.group(4)),
            "AP_1.0":    float(m.group(5)),
            "AP_1.5":    float(m.group(6)),
            "AP":        float(m.group(7)),
        }

    if len(cat_data) < 3:
        return None

    return {
        "condition_tag": condition_tag,
        "mAP_normal":    mAP_normal,
        "categories":    cat_data,
    }


def find_log_dir(suite_root: Path):
    """Auto-detect the most recent sbatch log dir for b0 corruption eval."""
    base = REPO_ROOT / "work_dirs" / "sbatch" / "corruption_eval_b0"
    if not base.is_dir():
        return None
    runs = sorted(base.iterdir(), reverse=True)
    for run in runs:
        logs = run / "logs"
        if logs.is_dir() and any(logs.glob("*.out")):
            return logs
    return None


# ---------------------------------------------------------------------------
# JSON loaders
# ---------------------------------------------------------------------------

def load_contradiction_metrics(suite_root: Path, condition: str) -> dict:
    path = suite_root / condition / "contradiction_metrics.json"
    if not path.exists():
        sys.exit(f"[consolidate_b0] Missing: {path}")
    d = json.loads(path.read_text())
    metrics = d.get("metrics", {})
    return {
        "stale_persistence_time_mean":         metrics.get("stale_persistence_time_mean"),
        "cumulative_stale_FP_polyline_length": metrics.get("cumulative_stale_false_positive_polyline_length"),
        "alpha_separation_summary":            metrics.get("alpha_separation_summary"),
    }


def load_alpha_stats(suite_root: Path, condition: str) -> dict:
    path = suite_root / condition / "alpha_stats.json"
    if not path.exists():
        sys.exit(f"[consolidate_b0] Missing: {path}")
    d = json.loads(path.read_text())
    return {
        "alpha_mean_affected":         d.get("alpha_mean_affected"),
        "alpha_mean_clean_recent":     d.get("alpha_mean_clean_recent"),
        "alpha_mean_preserved_recent": d.get("alpha_mean_preserved_recent"),
    }


def load_cmap_results(suite_root: Path, condition: str) -> dict:
    """Load cmap_results.json if it exists; return empty dict if not yet computed."""
    path = suite_root / condition / "cmap_results.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    result = {"mean_cMAP": d.get("mean_cMAP")}
    for cat in CATEGORIES:
        result[f"{cat}_cAP"] = d.get("category_cAP", {}).get(cat)
    return result


# ---------------------------------------------------------------------------
# Build tables
# ---------------------------------------------------------------------------

def build_rows(suite_root: Path, log_dir: Path):
    # Parse all .out files, keyed by condition_tag
    log_data: dict[str, dict] = {}
    for f in sorted(log_dir.glob("*.out")):
        parsed = parse_eval_log(f)
        if parsed:
            log_data[parsed["condition_tag"]] = parsed

    rows = []
    for cond in CONDITIONS:
        if cond not in log_data:
            print(f"[consolidate_b0] WARNING: No log data found for {cond}, skipping.")
            continue

        log = log_data[cond]
        parts = cond.split("_")
        mode = f"{parts[0]}_{parts[1]}"
        stale_offset = int(parts[2].replace("offset", ""))
        onset = int(parts[3].replace("onset", ""))

        row = {
            "model":         "b0",
            "condition_tag": cond,
            "mode":          mode,
            "stale_offset":  stale_offset,
            "onset":         onset,
        }

        row.update(load_contradiction_metrics(suite_root, cond))
        row.update(load_alpha_stats(suite_root, cond))
        row.update(load_cmap_results(suite_root, cond))

        cat_data = log["categories"]
        for cat in CATEGORIES:
            cd = cat_data.get(cat, {})
            row[f"{cat}_num_preds"] = cd.get("num_preds", "")
            row[f"{cat}_num_gts"]   = cd.get("num_gts", "")
            row[f"{cat}_AP_0.5"]    = cd.get("AP_0.5", "")
            row[f"{cat}_AP_1.0"]    = cd.get("AP_1.0", "")
            row[f"{cat}_AP_1.5"]    = cd.get("AP_1.5", "")
            row[f"{cat}_AP"]        = cd.get("AP", "")

        row["mAP"] = log["mAP_normal"]
        # mean_mAP_across_conditions is filled in after all rows are assembled
        rows.append(row)

    # Compute mean mAP and mean cMAP across all conditions
    if rows:
        mean_mAP = sum(float(r["mAP"]) for r in rows) / len(rows)
        cmap_vals = [r["mean_cMAP"] for r in rows if r.get("mean_cMAP") is not None]
        mean_cMAP_all = round(sum(cmap_vals) / len(cmap_vals), 6) if cmap_vals else None
        for r in rows:
            r["mean_mAP_across_conditions"] = round(mean_mAP, 6)
            r["mean_cMAP_across_conditions"] = mean_cMAP_all

    return rows


def build_corrupted_eval_rows(rows):
    """Generate rows in the same format as corrupted_eval_results_b1_b2.csv."""
    out = []
    for row in rows:
        for cat in CATEGORIES:
            out.append({
                "model":     row["model"],
                "mode":      row["mode"],
                "offset":    row["stale_offset"],
                "condition": row["condition_tag"],
                "category":  cat,
                "num_preds": row.get(f"{cat}_num_preds", ""),
                "num_gts":   row.get(f"{cat}_num_gts", ""),
                "AP_0_5":    row.get(f"{cat}_AP_0.5", ""),
                "AP_1_0":    row.get(f"{cat}_AP_1.0", ""),
                "AP_1_5":    row.get(f"{cat}_AP_1.5", ""),
                "AP":        row.get(f"{cat}_AP", ""),
                "mAP_normal": row["mAP"],
                "save_path": "",
            })
    return out


# ---------------------------------------------------------------------------
# Formatters / writers
# ---------------------------------------------------------------------------

def fmt(val, decimals=4) -> str:
    if val is None or val == "":
        return ""
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def write_wide_csv(rows, out_path: Path) -> None:
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=WIDE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in WIDE_COLUMNS})
    print(f"[consolidate_b0] Wrote {len(rows)} rows -> {out_path}")


def write_corrupted_eval_csv(rows, out_path: Path) -> None:
    fieldnames = ["model", "mode", "offset", "condition", "category",
                  "num_preds", "num_gts", "AP_0_5", "AP_1_0", "AP_1_5", "AP", "mAP_normal", "save_path"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in fieldnames})
    print(f"[consolidate_b0] Wrote {len(rows)} rows -> {out_path}")


def write_markdown(rows, out_path: Path) -> None:
    lines = []
    lines.append("# B0 Contradiction Suite Results\n")
    lines.append("Model: **B0** (original MapTracker checkpoint, temporal gate disabled)\n")
    if rows:
        mean_mAP = rows[0].get("mean_mAP_across_conditions", "")
        lines.append(f"Mean mAP across all 6 conditions: **{fmt(mean_mAP, 4)}**")
        mean_cmap_all = rows[0].get("mean_cMAP_across_conditions")
        if mean_cmap_all is not None:
            lines.append(f"Mean cMAP across all 6 conditions: **{fmt(mean_cmap_all, 4)}**")
        lines.append("")

    # --- Contradiction metrics table ---
    lines.append("## Contradiction Metrics\n")
    lines.append("| Condition | Mode | Offset | Stale Persist. (frames) | Cum. Stale FP Length | Alpha Sep. Summary | α_affected | α_clean_recent | α_preserved_recent |")
    lines.append("|-----------|------|--------|------------------------|---------------------|-------------------|-----------|----------------|-------------------|")
    for r in rows:
        lines.append(
            f"| {r['condition_tag']} "
            f"| {r['mode']} "
            f"| {r['stale_offset']} "
            f"| {fmt(r['stale_persistence_time_mean'], 2)} "
            f"| {fmt(r['cumulative_stale_FP_polyline_length'], 0)} "
            f"| {fmt(r['alpha_separation_summary'], 4)} "
            f"| {fmt(r['alpha_mean_affected'], 4)} "
            f"| {fmt(r['alpha_mean_clean_recent'], 4)} "
            f"| {fmt(r['alpha_mean_preserved_recent'], 4)} |"
        )
    lines.append("")

    # --- AP metrics table ---
    lines.append("## AP Metrics\n")
    lines.append("| Condition | Mode | Offset | **mAP** | Ped AP@0.5 | Ped AP@1.0 | Ped AP@1.5 | Ped AP | Div AP@0.5 | Div AP@1.0 | Div AP@1.5 | Div AP | Bnd AP@0.5 | Bnd AP@1.0 | Bnd AP@1.5 | Bnd AP |")
    lines.append("|-----------|------|--------|---------|-----------|-----------|-----------|--------|-----------|-----------|-----------|--------|-----------|-----------|-----------|--------|")
    for r in rows:
        lines.append(
            f"| {r['condition_tag']} "
            f"| {r['mode']} "
            f"| {r['stale_offset']} "
            f"| **{fmt(r['mAP'], 4)}** "
            f"| {fmt(r['ped_crossing_AP_0.5'], 4)} "
            f"| {fmt(r['ped_crossing_AP_1.0'], 4)} "
            f"| {fmt(r['ped_crossing_AP_1.5'], 4)} "
            f"| {fmt(r['ped_crossing_AP'], 4)} "
            f"| {fmt(r['divider_AP_0.5'], 4)} "
            f"| {fmt(r['divider_AP_1.0'], 4)} "
            f"| {fmt(r['divider_AP_1.5'], 4)} "
            f"| {fmt(r['divider_AP'], 4)} "
            f"| {fmt(r['boundary_AP_0.5'], 4)} "
            f"| {fmt(r['boundary_AP_1.0'], 4)} "
            f"| {fmt(r['boundary_AP_1.5'], 4)} "
            f"| {fmt(r['boundary_AP'], 4)} |"
        )
    lines.append("")

    # --- cMAP metrics table (only if cmap data is present) ---
    has_cmap = any(r.get("mean_cMAP") is not None for r in rows)
    if has_cmap:
        lines.append("## cMAP Metrics (Consistency mAP)\n")
        lines.append("| Condition | Mode | Offset | **mean_cMAP** | Ped cAP | Div cAP | Bnd cAP |")
        lines.append("|-----------|------|--------|-------------|---------|---------|---------|")
        for r in rows:
            lines.append(
                f"| {r['condition_tag']} "
                f"| {r['mode']} "
                f"| {r['stale_offset']} "
                f"| **{fmt(r.get('mean_cMAP'), 4)}** "
                f"| {fmt(r.get('ped_crossing_cAP'), 4)} "
                f"| {fmt(r.get('divider_cAP'), 4)} "
                f"| {fmt(r.get('boundary_cAP'), 4)} |"
            )
        lines.append("")
    else:
        lines.append("")
        lines.append("> **cMAP not yet computed.** Run `bash tools/experiments/submit_cmap_b0_all.sh --mail-user EMAIL` to compute cMAP for all conditions, then re-run this script.\n")
        lines.append("")
    lines.append("## Prediction Counts\n")
    lines.append("| Condition | Ped Preds | Ped GTs | Div Preds | Div GTs | Bnd Preds | Bnd GTs |")
    lines.append("|-----------|-----------|---------|-----------|---------|-----------|---------|")
    for r in rows:
        lines.append(
            f"| {r['condition_tag']} "
            f"| {r['ped_crossing_num_preds']} "
            f"| {r['ped_crossing_num_gts']} "
            f"| {r['divider_num_preds']} "
            f"| {r['divider_num_gts']} "
            f"| {r['boundary_num_preds']} "
            f"| {r['boundary_num_gts']} |"
        )
    lines.append("")

    lines.append("---\n")
    lines.append("**Notes**\n")
    lines.append("- **mAP** is computed on the corrupted test set for each condition independently.")
    lines.append("- **mean_mAP_across_conditions** is the mean mAP over all 6 corruption conditions.")
    lines.append("- **cMAP** (consistency mAP) measures temporal consistency of predictions across consecutive frames. Higher = more consistent tracking.")
    lines.append("- **mean_cMAP** for B0 reflects baseline temporal consistency WITHOUT a learned temporal gate.")
    lines.append("- The near-identical mAP values across all conditions confirm that B0 has **no temporal gate** — it does not distinguish stale from fresh memory, so the corrupted memory is treated as valid context, yet standard mAP remains stable. This is the core contradiction: high persistence + high mAP = the model outputs stale predictions confidently without detecting the contradiction.")
    lines.append("")
    lines.append("**Column Definitions**\n")
    lines.append("- **Mode**: `c_full` = full memory corruption; `c_tail` = tail-only corruption")
    lines.append("- **Offset**: stale memory offset (frames; 1/2/3)")
    lines.append("- **Stale Persist.**: mean frames stale corruption persists across scenes")
    lines.append("- **Cum. Stale FP Length**: cumulative stale false-positive polyline length (meters, proxy metric)")
    lines.append("- **Alpha Sep. Summary**: alpha separability summary (B0 has no gate so always near 0 or negative)")
    lines.append("- **AP@T**: average precision at threshold T meters; AP = mean(AP@0.5, AP@1.0, AP@1.5)")
    lines.append("- **mAP**: mean AP across ped_crossing, divider, boundary (computed on corrupted test set)")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"[consolidate_b0] Wrote Markdown -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=REPO_ROOT / "work_dirs/experiments/b0/corruption_suite/latest_onset0_trainmatched",
        help="Path to B0 corruption suite directory",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Path to sbatch logs dir (auto-detected if omitted)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "CurrentB1B2Results" / "b0",
        help="Output directory for CSV and Markdown files",
    )
    args = parser.parse_args()

    suite_root = args.suite_root.resolve()
    if not suite_root.is_dir():
        sys.exit(f"[consolidate_b0] suite-root not found: {suite_root}")

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = find_log_dir(suite_root)
        if log_dir is None:
            sys.exit("[consolidate_b0] Could not auto-detect log dir. Use --log-dir.")
    log_dir = log_dir.resolve()
    if not log_dir.is_dir():
        sys.exit(f"[consolidate_b0] log-dir not found: {log_dir}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[consolidate_b0] Suite root: {suite_root}")
    print(f"[consolidate_b0] Log dir:    {log_dir}")
    print(f"[consolidate_b0] Output dir: {out_dir}")
    print()

    rows = build_rows(suite_root, log_dir)
    if not rows:
        sys.exit("[consolidate_b0] No rows built — check that log files exist and are complete.")

    write_wide_csv(rows, out_dir / "contradiction_results_b0.csv")
    write_corrupted_eval_csv(build_corrupted_eval_rows(rows), out_dir / "corrupted_eval_results_b0.csv")
    write_markdown(rows, out_dir / "contradiction_results_b0_summary.md")

    # Print quick summary to stdout
    print()
    has_cmap = any(r.get("mean_cMAP") is not None for r in rows)
    print("=== B0 Corruption Suite Summary ===")
    header = f"{'Condition':<30} {'Mode':<8} {'Off':>3}  {'mAP':>8}  {'cMAP':>8}  {'Stale Persist':>13}  {'Alpha Sep':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        cmap_str = fmt(r.get("mean_cMAP"), 4) if has_cmap else "  N/A  "
        print(
            f"{r['condition_tag']:<30} {r['mode']:<8} {r['stale_offset']:>3}  "
            f"{fmt(r['mAP'], 4):>8}  {cmap_str:>8}  "
            f"{fmt(r['stale_persistence_time_mean'], 2):>13}  "
            f"{fmt(r['alpha_separation_summary'], 4):>10}"
        )
    if rows:
        print("-" * len(header))
        print(f"{'Mean mAP across conditions:':<45} {fmt(rows[0]['mean_mAP_across_conditions'], 4):>8}")
        if has_cmap and rows[0].get("mean_cMAP_across_conditions") is not None:
            print(f"{'Mean cMAP across conditions:':<45} {fmt(rows[0]['mean_cMAP_across_conditions'], 4):>8}")
        elif not has_cmap:
            print(f"  cMAP not yet computed. Run: bash tools/experiments/submit_cmap_b0_all.sh --mail-user EMAIL")
    print()
    print("[consolidate_b0] Done.")


if __name__ == "__main__":
    main()
