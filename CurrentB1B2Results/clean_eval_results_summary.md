# B1/B2 Clean Evaluation Results — iter_89148

Checkpoint: `iter_89148`  |  Condition: clean (no memory corruption)  |  Val scenes: 6019 frames

## Standard Map AP

| Model | mAP | Ped AP@0.5 | Ped AP@1.0 | Ped AP@1.5 | Ped AP | Div AP@0.5 | Div AP@1.0 | Div AP@1.5 | Div AP | Bnd AP@0.5 | Bnd AP@1.0 | Bnd AP@1.5 | Bnd AP |
|-------|-----|-----------|-----------|-----------|--------|-----------|-----------|-----------|--------|-----------|-----------|-----------|--------|
| **B1** | **0.7300** | 0.5997 | 0.7807 | 0.8566 | 0.7457 | 0.6086 | 0.7301 | 0.7942 | 0.7110 | 0.5891 | 0.7693 | 0.8415 | 0.7333 |
| **B2** | **0.7303** | 0.6216 | 0.8134 | 0.8677 | 0.7676 | 0.6061 | 0.7267 | 0.7764 | 0.7030 | 0.5807 | 0.7613 | 0.8189 | 0.7203 |

## Consistency Map AP (cMAP, cons_frames=5)

| Model | mean cMAP | Ped cAP@0.5 | Ped cAP@1.0 | Ped cAP@1.5 | Ped cAP | Div cAP@0.5 | Div cAP@1.0 | Div cAP@1.5 | Div cAP | Bnd cAP@0.5 | Bnd cAP@1.0 | Bnd cAP@1.5 | Bnd cAP |
|-------|-----------|------------|------------|------------|---------|------------|------------|------------|---------|------------|------------|------------|---------|
| **B1** | **0.6729** | 0.5829 | 0.7449 | 0.8113 | 0.7130 |  |  |  | 0.6333 |  |  |  | 0.6723 |
| **B2** | **0.6703** | 0.6042 | 0.7753 | 0.8170 | 0.7322 |  |  |  | 0.6147 |  |  |  | 0.6641 |

## Prediction Counts

| Model | Ped Preds | Ped GTs | Div Preds | Div GTs | Bnd Preds | Bnd GTs |
|-------|-----------|---------|-----------|---------|-----------|---------|
| **B1** | 91933 | 6922 | 273549 | 27332 | 284123 | 21050 |
| **B2** | 57391 | 6922 | 194178 | 27332 | 396474 | 21050 |

## Alpha Stats (clean memory)

| Model | α_mean_affected | α_mean_clean_recent | α_mean_preserved_recent |
|-------|-----------------|---------------------|------------------------|
| **B1** | 0.0000 | 0.9751 | 0.9751 |
| **B2** | 0.0000 | 0.9366 | 0.9366 |

## Run Metadata

| Model | run_id | checkpoint_tag | gpus | launcher | status |
|-------|--------|----------------|------|----------|--------|
| **B1** | b1_stage3_gpu4_short_trainonly | iter_89148 | 4 | slurm-step | success |
| **B2** | b2_stage3_gpu4_short_trainonly | iter_89148 | 4 | slurm-step | success |

---

**Column Definitions**

- **mAP**: mean AP across ped_crossing, divider, boundary (standard evaluation)
- **AP@T**: average precision at matching threshold T (meters); AP = mean of AP@0.5/1.0/1.5
- **mean cMAP**: consistency mean AP — evaluates temporal consistency of predictions across `cons_frames=5` consecutive frames
- **cAP@T**: consistency AP at threshold T
- **α_mean_clean_recent / α_mean_preserved_recent**: mean attention weight on clean/preserved recent memory; should be high for a healthy model (no corruption applied)
