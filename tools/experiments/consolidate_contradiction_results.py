#!/usr/bin/env python3
"""
Consolidate B1/B2 contradiction suite results into a wide-format CSV
and a human-readable Markdown summary table.

Usage (from any CWD):
    python tools/experiments/consolidate_contradiction_results.py [--results-root PATH]

Outputs (written to RESULTS_ROOT):
    contradiction_results_master.csv   – wide format, one row per model x condition
    contradiction_results_summary.md   – Markdown tables (B1 then B2)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # maptracker_v1/

DEFAULT_RESULTS_ROOT = REPO_ROOT / "CurrentB1B2Results"

CONDITIONS = [
    "c_full_offset1_onset0",
    "c_full_offset2_onset0",
    "c_full_offset3_onset0",
    "c_tail_offset1_onset0",
    "c_tail_offset2_onset0",
    "c_tail_offset3_onset0",
]

MODELS = ["b1", "b2"]

CATEGORIES = ["ped_crossing", "divider", "boundary"]

WIDE_COLUMNS = (
    ["model", "condition_tag", "mode", "stale_offset", "onset"]
    + ["stale_persistence_time_mean", "cumulative_stale_FP_polyline_length", "alpha_separation_summary"]
    + ["alpha_mean_affected", "alpha_mean_clean_recent", "alpha_mean_preserved_recent"]
    + [f"{cat}_num_preds"  for cat in CATEGORIES]
    + [f"{cat}_num_gts"    for cat in CATEGORIES]
    + [f"{cat}_AP_0.5"     for cat in CATEGORIES]
    + [f"{cat}_AP_1.0"     for cat in CATEGORIES]
    + [f"{cat}_AP_1.5"     for cat in CATEGORIES]
    + [f"{cat}_AP"         for cat in CATEGORIES]
    + ["mAP"]
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ap_data(results_root: Path) -> dict:
    """
    Load corrupted_eval_results_b1_b2.csv and pivot to wide format.
    Returns: {(model, condition_tag): {col: value, ...}}
    """
    csv_path = results_root / "corrupted_eval_results_b1_b2.csv"
    if not csv_path.exists():
        sys.exit(f"[consolidate] Missing file: {csv_path}")

    # keyed by (model, condition_tag) -> {category -> row_dict}
    raw: dict = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row["model"], row["condition"])
            raw.setdefault(key, {})[row["category"]] = row

    result = {}
    for (model, cond), cat_rows in raw.items():
        entry = {}
        # mAP is the same across categories for a given condition
        entry["mAP"] = float(list(cat_rows.values())[0]["mAP_normal"])
        for cat in CATEGORIES:
            r = cat_rows.get(cat, {})
            entry[f"{cat}_num_preds"] = int(r.get("num_preds", 0))
            entry[f"{cat}_num_gts"]   = int(r.get("num_gts", 0))
            entry[f"{cat}_AP_0.5"]    = float(r.get("AP_0_5", "nan"))
            entry[f"{cat}_AP_1.0"]    = float(r.get("AP_1_0", "nan"))
            entry[f"{cat}_AP_1.5"]    = float(r.get("AP_1_5", "nan"))
            entry[f"{cat}_AP"]        = float(r.get("AP", "nan"))
        result[(model, cond)] = entry
    return result


def load_suite_summary(results_root: Path, model: str) -> dict:
    """
    Load contradiction_suite_summary.json for a model.
    Returns: {condition_tag: {stale_persistence_time_mean, cumulative_stale_FP_polyline_length, alpha_separation_summary}}
    """
    path = results_root / f"{model}_contra_89148" / "contradiction_suite_summary.json"
    if not path.exists():
        sys.exit(f"[consolidate] Missing file: {path}")

    with open(path) as fh:
        data = json.load(fh)

    result = {}
    for cond_tag, cond_data in data.get("conditions", {}).items():
        metrics = cond_data.get("metrics", {})
        result[cond_tag] = {
            "stale_persistence_time_mean":           metrics.get("stale_persistence_time_mean"),
            "cumulative_stale_FP_polyline_length":   metrics.get("cumulative_stale_false_positive_polyline_length"),
            "alpha_separation_summary":              metrics.get("alpha_separation_summary"),
        }
    return result


def load_alpha_stats(results_root: Path, model: str, condition: str) -> dict:
    """
    Load alpha_stats.json for a specific model/condition.
    """
    path = results_root / f"{model}_contra_89148" / condition / "alpha_stats.json"
    if not path.exists():
        sys.exit(f"[consolidate] Missing file: {path}")

    with open(path) as fh:
        data = json.load(fh)
    return {
        "alpha_mean_affected":          data.get("alpha_mean_affected"),
        "alpha_mean_clean_recent":      data.get("alpha_mean_clean_recent"),
        "alpha_mean_preserved_recent":  data.get("alpha_mean_preserved_recent"),
    }


# ---------------------------------------------------------------------------
# Build master table
# ---------------------------------------------------------------------------

def build_rows(results_root: Path) -> list:
    ap_data = load_ap_data(results_root)

    rows = []
    for model in MODELS:
        suite_summary = load_suite_summary(results_root, model)
        for cond in CONDITIONS:
            # Parse condition tag -> mode, offset, onset
            # e.g. c_full_offset2_onset0  ->  mode=c_full, offset=2, onset=0
            parts = cond.split("_")
            # mode is first two parts joined: c_full or c_tail
            mode = f"{parts[0]}_{parts[1]}"
            stale_offset = int(parts[2].replace("offset", ""))
            onset = int(parts[3].replace("onset", ""))

            row = {
                "model":          model,
                "condition_tag":  cond,
                "mode":           mode,
                "stale_offset":   stale_offset,
                "onset":          onset,
            }

            # Suite summary metrics
            suite = suite_summary.get(cond, {})
            row["stale_persistence_time_mean"]         = suite.get("stale_persistence_time_mean")
            row["cumulative_stale_FP_polyline_length"] = suite.get("cumulative_stale_FP_polyline_length")
            row["alpha_separation_summary"]            = suite.get("alpha_separation_summary")

            # Alpha stats
            alpha = load_alpha_stats(results_root, model, cond)
            row.update(alpha)

            # AP metrics
            ap_key = (model, cond)
            if ap_key not in ap_data:
                sys.exit(f"[consolidate] No AP data found for key {ap_key}")
            row.update(ap_data[ap_key])

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def fmt(val, decimals=4) -> str:
    """Format a numeric value for display."""
    if val is None:
        return ""
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def write_csv(rows: list, out_path: Path) -> None:
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=WIDE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in WIDE_COLUMNS})
    print(f"[consolidate] Wrote {len(rows)} rows -> {out_path}")


def write_markdown(rows: list, out_path: Path) -> None:
    lines = []
    lines.append("# B1/B2 Contradiction Suite Results — iter_89148\n")
    lines.append(f"Checkpoint: `iter_89148`  |  Onset: 0  |  Scenes: ~121\n")

    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model]

        lines.append(f"\n## Model: {model.upper()}\n")

        # --- Contradiction metrics table ---
        lines.append("### Contradiction Metrics\n")
        lines.append("| Condition | Mode | Offset | Stale Persist. (frames) | Cum. Stale FP Length | Alpha Sep. Summary | α_mean_affected | α_mean_clean_recent | α_mean_preserved_recent |")
        lines.append("|-----------|------|--------|------------------------|---------------------|-------------------|-----------------|---------------------|------------------------|")
        for r in model_rows:
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
        lines.append("### AP Metrics\n")
        lines.append("| Condition | Mode | Offset | mAP | Ped AP@0.5 | Ped AP@1.0 | Ped AP@1.5 | Ped AP | Div AP@0.5 | Div AP@1.0 | Div AP@1.5 | Div AP | Bnd AP@0.5 | Bnd AP@1.0 | Bnd AP@1.5 | Bnd AP |")
        lines.append("|-----------|------|--------|-----|-----------|-----------|-----------|--------|-----------|-----------|-----------|--------|-----------|-----------|-----------|--------|")
        for r in model_rows:
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

        # --- Prediction counts table ---
        lines.append("### Prediction Counts\n")
        lines.append("| Condition | Ped Preds | Ped GTs | Div Preds | Div GTs | Bnd Preds | Bnd GTs |")
        lines.append("|-----------|-----------|---------|-----------|---------|-----------|---------|")
        for r in model_rows:
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

    # Footer with column definitions
    lines.append("---\n")
    lines.append("**Column Definitions**\n")
    lines.append("- **Mode**: `c_full` = full memory corruption; `c_tail` = tail-only corruption")
    lines.append("- **Offset**: stale memory offset (frames; 1/2/3)")
    lines.append("- **Stale Persist.**: mean number of frames stale corruption persists across all scenes")
    lines.append("- **Cum. Stale FP Length**: cumulative stale false-positive polyline length (proxy metric, meters)")
    lines.append("- **Alpha Sep. Summary**: overall alpha separability summary (higher = model better distinguishes stale from fresh memory)")
    lines.append("- **α_mean_affected**: mean alpha weight on corrupted/affected memory slots")
    lines.append("- **α_mean_clean_recent**: mean alpha weight on clean recent memory slots")
    lines.append("- **α_mean_preserved_recent**: mean alpha weight on preserved recent memory slots")
    lines.append("- **AP@T**: average precision at matching threshold T (meters); AP = mean of AP@0.5/1.0/1.5")
    lines.append("- **mAP**: mean AP across ped_crossing, divider, boundary categories")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[consolidate] Wrote Markdown -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Path to CurrentB1B2Results directory (default: {DEFAULT_RESULTS_ROOT})",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    if not results_root.is_dir():
        sys.exit(f"[consolidate] results-root not found: {results_root}")

    print(f"[consolidate] Reading from: {results_root}")

    rows = build_rows(results_root)

    csv_out = results_root / "contradiction_results_master.csv"
    md_out  = results_root / "contradiction_results_summary.md"

    write_csv(rows, csv_out)
    write_markdown(rows, md_out)

    print("[consolidate] Done.")


if __name__ == "__main__":
    main()
