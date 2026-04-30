#!/usr/bin/env python3
"""
Consolidate B1/B2 clean evaluation results into a wide-format CSV
and a human-readable Markdown summary table.

Parses the eval logs to extract:
  - Standard map AP  (per-category AP@0.5/1.0/1.5 + AP + mAP, num_preds, num_gts)
  - Consistency map AP / cMAP  (per-category AP@0.5/1.0/1.5 + mean cMAP)
  - Alpha stats  (alpha_mean_affected, alpha_mean_clean_recent, alpha_mean_preserved_recent)
  - Run metadata  (run_id, checkpoint_tag, gpus, launcher, status)

Usage (from any CWD):
    python tools/experiments/consolidate_clean_eval_results.py [--results-root PATH]

Outputs (written to RESULTS_ROOT):
    clean_eval_results_master.csv    – wide format, one row per model
    clean_eval_results_summary.md   – Markdown table
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # maptracker_v1/

DEFAULT_RESULTS_ROOT = REPO_ROOT / "CurrentB1B2Results"

MODELS = ["b1", "b2"]
CATEGORIES = ["ped_crossing", "divider", "boundary"]

WIDE_COLUMNS = (
    ["model", "run_id", "checkpoint_tag", "condition_tag", "gpus", "launcher", "status"]
    + ["alpha_mean_affected", "alpha_mean_clean_recent", "alpha_mean_preserved_recent"]
    # Standard map AP
    + [f"{cat}_num_preds" for cat in CATEGORIES]
    + [f"{cat}_num_gts"   for cat in CATEGORIES]
    + [f"{cat}_AP_0.5"    for cat in CATEGORIES]
    + [f"{cat}_AP_1.0"    for cat in CATEGORIES]
    + [f"{cat}_AP_1.5"    for cat in CATEGORIES]
    + [f"{cat}_AP"        for cat in CATEGORIES]
    + ["mAP"]
    # Consistency map AP (cMAP)
    + [f"{cat}_cAP_0.5"   for cat in CATEGORIES]
    + [f"{cat}_cAP_1.0"   for cat in CATEGORIES]
    + [f"{cat}_cAP_1.5"   for cat in CATEGORIES]
    + [f"{cat}_cAP"       for cat in CATEGORIES]
    + ["mean_cMAP"]
)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_log(log_path: Path) -> dict:
    """
    Parse run_b1_b2_deferred_eval.log and extract:
      - standard_ap: {category: {num_preds, num_gts, AP_0.5, AP_1.0, AP_1.5, AP}}
      - mAP: float
      - cmap: {category: {cAP_0.5, cAP_1.0, cAP_1.5, cAP}}
      - mean_cMAP: float
    """
    if not log_path.exists():
        sys.exit(f"[consolidate_clean] Missing log: {log_path}")

    text = log_path.read_text(errors="replace")

    result = {}

    # -----------------------------------------------------------------------
    # Standard AP table  (ASCII table with | separators)
    # Example line:
    #   | ped_crossing |   91933   |   6922  | 0.5997 | 0.7807 | 0.8566 | 0.7457 |
    # -----------------------------------------------------------------------
    std_ap = {}
    ap_row_re = re.compile(
        r"\|\s*(ped_crossing|divider|boundary)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
        r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    )
    for m in ap_row_re.finditer(text):
        cat = m.group(1)
        std_ap[cat] = {
            "num_preds": int(m.group(2)),
            "num_gts":   int(m.group(3)),
            "AP_0.5":    float(m.group(4)),
            "AP_1.0":    float(m.group(5)),
            "AP_1.5":    float(m.group(6)),
            "AP":        float(m.group(7)),
        }
    result["standard_ap"] = std_ap

    # mAP_normal — keep as raw string to preserve trailing zeros (e.g. "0.7300")
    mmap_m = re.search(r"mAP_normal\s*=\s*([\d.]+)", text)
    result["mAP"] = mmap_m.group(1) if mmap_m else None

    # -----------------------------------------------------------------------
    # cMAP  — three separate dict lines + "mean AP" line
    # Example:
    #   {'ped_crossing': {'AP@0.5': 0.5828...}, 'divider': ...}
    #   {'ped_crossing': {'AP@1.0': 0.7448...}, ...}
    #   {'ped_crossing': {'AP@1.5': 0.8113...}, ...}
    #   Category mean AP {'ped_crossing': 0.7130..., ...}
    #   mean AP  0.6728...
    # -----------------------------------------------------------------------
    # Find the three consecutive dict lines by looking for AP@0.5 / AP@1.0 / AP@1.5
    cmap: dict = {cat: {} for cat in CATEGORIES}

    def parse_cmap_line(pattern: str, key: str):
        """Find the line matching pattern and populate cmap[cat][key]."""
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            return
        snippet = m.group(0)
        for cat in CATEGORIES:
            cat_m = re.search(
                re.escape(f"'{cat}'") + r"\s*:\s*\{[^}]*" + re.escape(f"'AP@{key}'") + r"\s*:\s*([\d.]+)",
                snippet,
            )
            if cat_m:
                cmap[cat][f"cAP_{key.replace('.', '.')}"] = float(cat_m.group(1))

    parse_cmap_line(r"\{[^}]*'AP@0\.5'[^}]*\}", "0.5")
    parse_cmap_line(r"\{[^}]*'AP@1\.0'[^}]*\}", "1.0")
    parse_cmap_line(r"\{[^}]*'AP@1\.5'[^}]*\}", "1.5")

    # Category mean AP line  ->  per-category cAP
    cat_mean_m = re.search(r"Category mean AP\s*(\{[^}]+\})", text)
    if cat_mean_m:
        cat_str = cat_mean_m.group(1)
        for cat in CATEGORIES:
            v_m = re.search(re.escape(f"'{cat}'") + r"\s*:\s*([\d.]+)", cat_str)
            if v_m:
                cmap[cat]["cAP"] = float(v_m.group(1))

    result["cmap"] = cmap

    # mean cMAP
    mean_cmap_m = re.search(r"mean AP\s+([\d.]+)", text)
    result["mean_cMAP"] = float(mean_cmap_m.group(1)) if mean_cmap_m else None

    return result


def load_alpha_stats(results_root: Path, model: str) -> dict:
    path = results_root / f"{model}_eval_89148" / "clean" / "alpha_stats.json"
    if not path.exists():
        sys.exit(f"[consolidate_clean] Missing: {path}")
    with open(path) as fh:
        data = json.load(fh)
    return {
        "alpha_mean_affected":         data.get("alpha_mean_affected"),
        "alpha_mean_clean_recent":     data.get("alpha_mean_clean_recent"),
        "alpha_mean_preserved_recent": data.get("alpha_mean_preserved_recent"),
    }


def load_manifest(results_root: Path, model: str) -> dict:
    path = results_root / f"{model}_eval_89148" / "clean" / "manifest.json"
    if not path.exists():
        sys.exit(f"[consolidate_clean] Missing: {path}")
    with open(path) as fh:
        data = json.load(fh)
    return {
        "run_id":          data.get("run_id", ""),
        "checkpoint_tag":  data.get("checkpoint_tag", ""),
        "condition_tag":   data.get("condition_tag", "clean"),
        "gpus":            data.get("gpus", ""),
        "launcher":        data.get("launcher", ""),
        "status":          data.get("status", ""),
    }


# ---------------------------------------------------------------------------
# Build master table
# ---------------------------------------------------------------------------

def build_rows(results_root: Path) -> list:
    rows = []
    for model in MODELS:
        log_path = results_root / f"{model}_eval_89148" / "clean" / "logs" / "run_b1_b2_deferred_eval.log"
        log_data  = parse_log(log_path)
        alpha     = load_alpha_stats(results_root, model)
        manifest  = load_manifest(results_root, model)

        row = {"model": model}
        row.update(manifest)
        row.update(alpha)

        std_ap = log_data.get("standard_ap", {})
        for cat in CATEGORIES:
            cat_data = std_ap.get(cat, {})
            row[f"{cat}_num_preds"] = cat_data.get("num_preds", "")
            row[f"{cat}_num_gts"]   = cat_data.get("num_gts", "")
            row[f"{cat}_AP_0.5"]    = cat_data.get("AP_0.5", "")
            row[f"{cat}_AP_1.0"]    = cat_data.get("AP_1.0", "")
            row[f"{cat}_AP_1.5"]    = cat_data.get("AP_1.5", "")
            row[f"{cat}_AP"]        = cat_data.get("AP", "")

        row["mAP"] = log_data.get("mAP", "")

        cmap = log_data.get("cmap", {})
        for cat in CATEGORIES:
            cat_data = cmap.get(cat, {})
            row[f"{cat}_cAP_0.5"] = cat_data.get("cAP_0.5", "")
            row[f"{cat}_cAP_1.0"] = cat_data.get("cAP_1.0", "")
            row[f"{cat}_cAP_1.5"] = cat_data.get("cAP_1.5", "")
            row[f"{cat}_cAP"]     = cat_data.get("cAP", "")

        row["mean_cMAP"] = log_data.get("mean_cMAP", "")

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def fmt(val, decimals=4) -> str:
    if val is None or val == "":
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
    print(f"[consolidate_clean] Wrote {len(rows)} rows -> {out_path}")


def write_markdown(rows: list, out_path: Path) -> None:
    lines = []
    lines.append("# B1/B2 Clean Evaluation Results — iter_89148\n")
    lines.append("Checkpoint: `iter_89148`  |  Condition: clean (no memory corruption)  |  Val scenes: 6019 frames\n")

    # --- Standard map AP ---
    lines.append("## Standard Map AP\n")
    lines.append("| Model | mAP | Ped AP@0.5 | Ped AP@1.0 | Ped AP@1.5 | Ped AP | Div AP@0.5 | Div AP@1.0 | Div AP@1.5 | Div AP | Bnd AP@0.5 | Bnd AP@1.0 | Bnd AP@1.5 | Bnd AP |")
    lines.append("|-------|-----|-----------|-----------|-----------|--------|-----------|-----------|-----------|--------|-----------|-----------|-----------|--------|")
    for r in rows:
        lines.append(
            f"| **{r['model'].upper()}** "
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

    # --- Consistency map AP ---
    lines.append("## Consistency Map AP (cMAP, cons_frames=5)\n")
    lines.append("| Model | mean cMAP | Ped cAP@0.5 | Ped cAP@1.0 | Ped cAP@1.5 | Ped cAP | Div cAP@0.5 | Div cAP@1.0 | Div cAP@1.5 | Div cAP | Bnd cAP@0.5 | Bnd cAP@1.0 | Bnd cAP@1.5 | Bnd cAP |")
    lines.append("|-------|-----------|------------|------------|------------|---------|------------|------------|------------|---------|------------|------------|------------|---------|")
    for r in rows:
        lines.append(
            f"| **{r['model'].upper()}** "
            f"| **{fmt(r['mean_cMAP'], 4)}** "
            f"| {fmt(r['ped_crossing_cAP_0.5'], 4)} "
            f"| {fmt(r['ped_crossing_cAP_1.0'], 4)} "
            f"| {fmt(r['ped_crossing_cAP_1.5'], 4)} "
            f"| {fmt(r['ped_crossing_cAP'], 4)} "
            f"| {fmt(r['divider_cAP_0.5'], 4)} "
            f"| {fmt(r['divider_cAP_1.0'], 4)} "
            f"| {fmt(r['divider_cAP_1.5'], 4)} "
            f"| {fmt(r['divider_cAP'], 4)} "
            f"| {fmt(r['boundary_cAP_0.5'], 4)} "
            f"| {fmt(r['boundary_cAP_1.0'], 4)} "
            f"| {fmt(r['boundary_cAP_1.5'], 4)} "
            f"| {fmt(r['boundary_cAP'], 4)} |"
        )

    lines.append("")

    # --- Prediction counts ---
    lines.append("## Prediction Counts\n")
    lines.append("| Model | Ped Preds | Ped GTs | Div Preds | Div GTs | Bnd Preds | Bnd GTs |")
    lines.append("|-------|-----------|---------|-----------|---------|-----------|---------|")
    for r in rows:
        lines.append(
            f"| **{r['model'].upper()}** "
            f"| {r['ped_crossing_num_preds']} "
            f"| {r['ped_crossing_num_gts']} "
            f"| {r['divider_num_preds']} "
            f"| {r['divider_num_gts']} "
            f"| {r['boundary_num_preds']} "
            f"| {r['boundary_num_gts']} |"
        )

    lines.append("")

    # --- Alpha stats ---
    lines.append("## Alpha Stats (clean memory)\n")
    lines.append("| Model | α_mean_affected | α_mean_clean_recent | α_mean_preserved_recent |")
    lines.append("|-------|-----------------|---------------------|------------------------|")
    for r in rows:
        lines.append(
            f"| **{r['model'].upper()}** "
            f"| {fmt(r['alpha_mean_affected'], 4)} "
            f"| {fmt(r['alpha_mean_clean_recent'], 4)} "
            f"| {fmt(r['alpha_mean_preserved_recent'], 4)} |"
        )

    lines.append("")

    # --- Run metadata ---
    lines.append("## Run Metadata\n")
    lines.append("| Model | run_id | checkpoint_tag | gpus | launcher | status |")
    lines.append("|-------|--------|----------------|------|----------|--------|")
    for r in rows:
        lines.append(
            f"| **{r['model'].upper()}** "
            f"| {r['run_id']} "
            f"| {r['checkpoint_tag']} "
            f"| {r['gpus']} "
            f"| {r['launcher']} "
            f"| {r['status']} |"
        )

    lines.append("")
    lines.append("---\n")
    lines.append("**Column Definitions**\n")
    lines.append("- **mAP**: mean AP across ped_crossing, divider, boundary (standard evaluation)")
    lines.append("- **AP@T**: average precision at matching threshold T (meters); AP = mean of AP@0.5/1.0/1.5")
    lines.append("- **mean cMAP**: consistency mean AP — evaluates temporal consistency of predictions across `cons_frames=5` consecutive frames")
    lines.append("- **cAP@T**: consistency AP at threshold T")
    lines.append("- **α_mean_clean_recent / α_mean_preserved_recent**: mean attention weight on clean/preserved recent memory; should be high for a healthy model (no corruption applied)")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[consolidate_clean] Wrote Markdown -> {out_path}")


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
        sys.exit(f"[consolidate_clean] results-root not found: {results_root}")

    print(f"[consolidate_clean] Reading from: {results_root}")

    rows = build_rows(results_root)

    csv_out = results_root / "clean_eval_results_master.csv"
    md_out  = results_root / "clean_eval_results_summary.md"

    write_csv(rows, csv_out)
    write_markdown(rows, md_out)

    print("[consolidate_clean] Done.")


if __name__ == "__main__":
    main()
