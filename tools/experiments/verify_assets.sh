#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: verify_assets.sh --config CFG --checkpoint CKPT --work-root DIR [options]

Required:
  --config PATH              Config file path.
  --checkpoint PATH          Checkpoint path.
  --work-root DIR            Root output directory (must exist or be creatable).

Optional:
  --run-dir DIR              Exact run dir to verify/create checks against.
  --allow-nonempty-run-dir   Permit reusing an existing populated run dir.
  --dataset-root DIR         Optional dataset root check.
  --map-file PATH            Optional map/annotation file check (repeatable).
  --require-path PATH        Optional additional required path (repeatable).
  --dry-run                  Print checks only; do not create directories/files.
  -h, --help                 Show this help.
USAGE
}

CONFIG=""
CHECKPOINT=""
WORK_ROOT=""
RUN_DIR=""
DATASET_ROOT=""
DRY_RUN=0
ALLOW_NONEMPTY_RUN_DIR=0

declare -a MAP_FILES=()
declare -a EXTRA_PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"; shift 2 ;;
    --checkpoint)
      CHECKPOINT="$2"; shift 2 ;;
    --work-root)
      WORK_ROOT="$2"; shift 2 ;;
    --run-dir)
      RUN_DIR="$2"; shift 2 ;;
    --allow-nonempty-run-dir)
      ALLOW_NONEMPTY_RUN_DIR=1; shift ;;
    --dataset-root)
      DATASET_ROOT="$2"; shift 2 ;;
    --map-file)
      MAP_FILES+=("$2"); shift 2 ;;
    --require-path)
      EXTRA_PATHS+=("$2"); shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[verify_assets] Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$CONFIG" || -z "$CHECKPOINT" || -z "$WORK_ROOT" ]]; then
  echo "[verify_assets] --config, --checkpoint, and --work-root are required." >&2
  usage
  exit 2
fi

fail() {
  echo "[verify_assets] ERROR: $*" >&2
  exit 1
}

check_file() {
  local f="$1"
  [[ -f "$f" ]] || fail "Missing file: $f"
  [[ -r "$f" ]] || fail "Unreadable file: $f"
  echo "[verify_assets] OK file: $f"
}

check_path_exists() {
  local p="$1"
  [[ -e "$p" ]] || fail "Missing path: $p"
  echo "[verify_assets] OK path: $p"
}

check_writable_dir() {
  local d="$1"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[verify_assets] DRY-RUN writable dir check: $d"
    return 0
  fi
  mkdir -p "$d"
  local probe="$d/.verify_write_test_$$"
  : > "$probe"
  rm -f "$probe"
  echo "[verify_assets] OK writable dir: $d"
}

check_file "$CONFIG"
check_file "$CHECKPOINT"

if [[ -n "$DATASET_ROOT" ]]; then
  check_path_exists "$DATASET_ROOT"
fi

for f in "${MAP_FILES[@]}"; do
  check_file "$f"
done

for p in "${EXTRA_PATHS[@]}"; do
  check_path_exists "$p"
done

check_writable_dir "$WORK_ROOT"

if [[ -n "$RUN_DIR" ]]; then
  if [[ -e "$RUN_DIR" ]]; then
    if [[ -d "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]] && [[ $ALLOW_NONEMPTY_RUN_DIR -eq 0 ]]; then
      fail "Run dir already exists and is non-empty: $RUN_DIR"
    elif [[ ! -d "$RUN_DIR" ]]; then
      fail "Run dir path exists but is not a directory: $RUN_DIR"
    fi
  fi
  check_writable_dir "$(dirname "$RUN_DIR")"
  if [[ $ALLOW_NONEMPTY_RUN_DIR -eq 1 ]]; then
    echo "[verify_assets] OK run-dir reuse check: $RUN_DIR"
  else
    echo "[verify_assets] OK run-dir collision check: $RUN_DIR"
  fi
fi

summary_json=""
if [[ -n "$RUN_DIR" ]]; then
  summary_json="$RUN_DIR/preflight.json"
else
  summary_json="$WORK_ROOT/preflight.json"
fi
if [[ $DRY_RUN -eq 1 ]]; then
  summary_json="${summary_json%.json}_dryrun.json"
fi

if [[ $DRY_RUN -eq 0 ]]; then
  mkdir -p "$(dirname "$summary_json")"
  map_joined=""
  extra_joined=""
  if [[ ${#MAP_FILES[@]} -gt 0 ]]; then
    map_joined="$(printf "%s\n" "${MAP_FILES[@]}")"
  fi
  if [[ ${#EXTRA_PATHS[@]} -gt 0 ]]; then
    extra_joined="$(printf "%s\n" "${EXTRA_PATHS[@]}")"
  fi
  CONFIG="$CONFIG" CHECKPOINT="$CHECKPOINT" WORK_ROOT="$WORK_ROOT" RUN_DIR="$RUN_DIR" \
  DATASET_ROOT="$DATASET_ROOT" MAP_JOINED="$map_joined" EXTRA_JOINED="$extra_joined" \
  SUMMARY_JSON="$summary_json" python - <<'PY'
import json
import os

def split_lines(value):
    if not value:
        return []
    return [v for v in value.splitlines() if v]

payload = {
    "config": os.environ["CONFIG"],
    "checkpoint": os.environ["CHECKPOINT"],
    "work_root": os.environ["WORK_ROOT"],
    "run_dir": os.environ["RUN_DIR"],
    "dataset_root": os.environ["DATASET_ROOT"],
    "map_files": split_lines(os.environ.get("MAP_JOINED", "")),
    "extra_paths": split_lines(os.environ.get("EXTRA_JOINED", "")),
    "dry_run": False,
}
with open(os.environ["SUMMARY_JSON"], "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
  echo "[verify_assets] Wrote preflight: $summary_json"
else
  echo "[verify_assets] DRY-RUN complete; preflight would be written to: $summary_json"
fi
