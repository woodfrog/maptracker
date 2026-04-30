# B0 Contradiction Suite Results

Model: **B0** (original MapTracker checkpoint, temporal gate disabled)

Mean mAP across all 6 conditions: **0.7651**

## Contradiction Metrics

| Condition | Mode | Offset | Stale Persist. (frames) | Cum. Stale FP Length | Alpha Sep. Summary | α_affected | α_clean_recent | α_preserved_recent |
|-----------|------|--------|------------------------|---------------------|-------------------|-----------|----------------|-------------------|
| c_full_offset1_onset0 | c_full | 1 | 40.13 | 1677633 | -0.3796 | 0.8945 | 0.0000 | 0.5149 |
| c_full_offset2_onset0 | c_full | 2 | 40.13 | 1675859 | -0.2613 | 0.8952 | 0.0000 | 0.6338 |
| c_full_offset3_onset0 | c_full | 3 | 40.13 | 1678001 | -0.1334 | 0.8387 | 0.0000 | 0.7053 |
| c_tail_offset1_onset0 | c_tail | 1 | 40.13 | 1675324 | 0.1575 | 0.8176 | 0.0000 | 0.9751 |
| c_tail_offset2_onset0 | c_tail | 2 | 40.13 | 1676798 | 0.1381 | 0.8370 | 0.0000 | 0.9751 |
| c_tail_offset3_onset0 | c_tail | 3 | 40.13 | 1675489 | 0.5933 | 0.3818 | 0.0000 | 0.9751 |

## AP Metrics

| Condition | Mode | Offset | **mAP** | Ped AP@0.5 | Ped AP@1.0 | Ped AP@1.5 | Ped AP | Div AP@0.5 | Div AP@1.0 | Div AP@1.5 | Div AP | Bnd AP@0.5 | Bnd AP@1.0 | Bnd AP@1.5 | Bnd AP |
|-----------|------|--------|---------|-----------|-----------|-----------|--------|-----------|-----------|-----------|--------|-----------|-----------|-----------|--------|
| c_full_offset1_onset0 | c_full | 1 | **0.7649** | 0.6556 | 0.8428 | 0.8976 | 0.7987 | 0.6499 | 0.7704 | 0.8284 | 0.7495 | 0.6024 | 0.7834 | 0.8531 | 0.7463 |
| c_full_offset2_onset0 | c_full | 2 | **0.7648** | 0.6541 | 0.8429 | 0.8977 | 0.7983 | 0.6485 | 0.7699 | 0.8280 | 0.7488 | 0.6039 | 0.7841 | 0.8540 | 0.7473 |
| c_full_offset3_onset0 | c_full | 3 | **0.7646** | 0.6536 | 0.8428 | 0.8972 | 0.7979 | 0.6493 | 0.7699 | 0.8283 | 0.7492 | 0.6032 | 0.7833 | 0.8538 | 0.7468 |
| c_tail_offset1_onset0 | c_tail | 1 | **0.7654** | 0.6553 | 0.8432 | 0.8978 | 0.7988 | 0.6509 | 0.7709 | 0.8289 | 0.7502 | 0.6036 | 0.7836 | 0.8541 | 0.7471 |
| c_tail_offset2_onset0 | c_tail | 2 | **0.7655** | 0.6552 | 0.8437 | 0.8980 | 0.7990 | 0.6508 | 0.7708 | 0.8291 | 0.7502 | 0.6040 | 0.7838 | 0.8541 | 0.7473 |
| c_tail_offset3_onset0 | c_tail | 3 | **0.7655** | 0.6555 | 0.8438 | 0.8979 | 0.7991 | 0.6500 | 0.7705 | 0.8287 | 0.7497 | 0.6045 | 0.7842 | 0.8545 | 0.7477 |


> **cMAP not yet computed.** Run `bash tools/experiments/submit_cmap_b0_all.sh --mail-user EMAIL` to compute cMAP for all conditions, then re-run this script.


## Prediction Counts

| Condition | Ped Preds | Ped GTs | Div Preds | Div GTs | Bnd Preds | Bnd GTs |
|-----------|-----------|---------|-----------|---------|-----------|---------|
| c_full_offset1_onset0 | 175341 | 6922 | 292869 | 27332 | 183000 | 21050 |
| c_full_offset2_onset0 | 175380 | 6922 | 292940 | 27332 | 182919 | 21050 |
| c_full_offset3_onset0 | 175496 | 6922 | 292735 | 27332 | 182971 | 21050 |
| c_tail_offset1_onset0 | 176224 | 6922 | 292165 | 27332 | 182740 | 21050 |
| c_tail_offset2_onset0 | 176135 | 6922 | 292245 | 27332 | 182764 | 21050 |
| c_tail_offset3_onset0 | 176171 | 6922 | 292111 | 27332 | 182862 | 21050 |

---

**Notes**

- **mAP** is computed on the corrupted test set for each condition independently.
- **mean_mAP_across_conditions** is the mean mAP over all 6 corruption conditions.
- **cMAP** (consistency mAP) measures temporal consistency of predictions across consecutive frames. Higher = more consistent tracking.
- **mean_cMAP** for B0 reflects baseline temporal consistency WITHOUT a learned temporal gate.
- The near-identical mAP values across all conditions confirm that B0 has **no temporal gate** — it does not distinguish stale from fresh memory, so the corrupted memory is treated as valid context, yet standard mAP remains stable. This is the core contradiction: high persistence + high mAP = the model outputs stale predictions confidently without detecting the contradiction.

**Column Definitions**

- **Mode**: `c_full` = full memory corruption; `c_tail` = tail-only corruption
- **Offset**: stale memory offset (frames; 1/2/3)
- **Stale Persist.**: mean frames stale corruption persists across scenes
- **Cum. Stale FP Length**: cumulative stale false-positive polyline length (meters, proxy metric)
- **Alpha Sep. Summary**: alpha separability summary (B0 has no gate so always near 0 or negative)
- **AP@T**: average precision at threshold T meters; AP = mean(AP@0.5, AP@1.0, AP@1.5)
- **mAP**: mean AP across ped_crossing, divider, boundary (computed on corrupted test set)
