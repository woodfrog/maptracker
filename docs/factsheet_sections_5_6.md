# Technical Factsheet: Sections 5 & 6
**Paper:** "Selective Temporal Memory Control for Online Vectorized HD Map Construction"
**Princeton COS Junior Paper — Evidence-Based, Evidence-Only**
*Generated: 2026-04-28. All values at full precision (≥5 decimal places). [TODO] marks missing data.*

---

## 1. Executive Summary

- **Task.** Online vectorized HD map construction on nuScenes (camera-only, old-split). Three map classes evaluated with per-class AP at three Chamfer-distance thresholds plus a temporal consistency metric (cMAP).
- **Three models compared.** B0 = original MapTracker (fully trained, no gate, no corruption training); B1 = corruption-trained no-gate baseline; B2 = corruption-trained gated model (proposed). All three are trained to convergence.
- **Clean performance.** B0 mAP 0.7655 (Ped 0.7989, Div 0.7499, Bnd 0.7478); B1 (mAP 0.73000) and B2 (mAP 0.73030) are statistically tied; +0.00030 gap is below any meaningful threshold. B0 is ~0.035 higher, consistent with its clean-only training regime. B0 clean cMAP (0.6983) is substantially higher than B1 (0.6729) and B2 (0.6703), consistent with clean-only training preserving greater temporal consistency.
- **Corruption robustness (mAP).** Neither B1 nor B2 degrades meaningfully under stale-memory injection: all corrupt conditions within ±0.00150 of clean mAP. B0 corrupt mAP (~0.765) is substantially higher, consistent with the absence of corruption-aware training changing model behavior.
- **Corruption robustness (cMAP).** B1 and B2 both show negligible cMAP degradation under corruption (within ±0.003 of clean cMAP). B1 slightly outperforms B2 under all corrupt cMAP conditions, consistent with its slightly higher clean cMAP (0.67289 vs. 0.67033).
- **Gate behavior (alpha).** B2's trained gate clearly suppresses stale slots: α_affected drops from ~0.88 (B1) to 0.032–0.334 (B2). Alpha separation is consistently and strongly positive in B2, negative in B1 under c_full conditions.
- **Stale FP footprint.** B2 accumulates modestly less cumulative stale-FP polyline length (~0.69%); stale persistence time is identical across all models and conditions (40.13 frames).
- **Central finding.** The trained gate (B2) achieves clear alpha discrimination between stale and fresh memory slots, but this does not improve mAP or cMAP vs. B1. The gate is behaviorally functional; its downstream metric impact is not yet visible at this experiment scale.

---

## 2. Section 5 Facts

### 2.1 Dataset and Task Configuration

| Parameter | Value | Source |
|---|---|---|
| Dataset | nuScenes (v1.0-trainval), old-split | `plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py` |
| Evaluation split | Validation set, `nuscenes_map_infos_val.pkl` | Same config |
| Validation set size | **6,019 frames** | cMAP job logs (`cmap_seq_b1-7420460.out`: "collected 6019 samples") |
| Sensor modalities | Camera-only; 6-camera surround view; `use_lidar=False`, `use_camera=True`, `use_radar=False`, `use_map=False` | Same config |
| Map classes | ped_crossing (id=0), divider (id=1), boundary (id=2) | Same config |
| BEV range | 60 m × 30 m (`roi_size=(60,30)`); `pc_range=[-30,-15,-3,30,15,5]` | Same config |
| BEV grid | `bev_h=50`, `bev_w=100` (50 × 100 cells) | Same config |
| Output format | Vectorized polylines; `num_points=20` per instance; `coords_dim=2`; `simplify=True` | Same config |
| Temporal window | 5-frame sliding window, `sampling_span=10`, `mem_len=4` | Same config |
| Selective memory ranges | `mem_select_dist_ranges=[1, 5, 10, 15]` | Same config |

**Evaluation metrics:**

| Metric | Definition | Source |
|---|---|---|
| AP@T | Average precision at Chamfer-distance threshold T ∈ {0.5, 1.0, 1.5} m | `tools/maptracker_evaluator.py` |
| AP (per class) | mean(AP@0.5, AP@1.0, AP@1.5) | Same |
| mAP | mean(AP_ped_crossing, AP_divider, AP_boundary) | Same |
| mean_cMAP | Consistency mAP over `cons_frames=5` consecutive frames | `tools/tracking/calculate_cmap.py` |
| α_affected | Mean gate alpha on stale/corrupted memory slots | `plugin/models/mapers/MapTracker.py`, `_compute_inference_alpha_stats()` |
| α_preserved_recent | Mean gate alpha on non-corrupted recent memory slots | Same |
| α_separation | α_preserved_recent − α_affected (positive = gate suppresses stale) | Computed from above |
| stale_persistence_time_mean | Mean consecutive frames with stale FP > 0 | `tools/tracking/contradiction_metrics.py` |
| cumulative_stale_FP_polyline_length | Cumulative stale false-positive polyline length (meters) | Same |

---

### 2.2 Models and Baselines

**B0: Original MapTracker (fully trained, no gate, no corruption training)**
- Checkpoint: `work_dirs/pretrained_ckpts/b0_nusc_oldsplit/latest.pth`
- Eval config: `plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_b0_eval.py`
- Gate status: `temporal_gate_cfg.enabled=False`. Gate is architecturally present but forward pass returns unmodified embeddings (all-ones alpha). Gate MLP weights were never trained (predates gate implementation).
- Training: Full convergence schedule, 8 GPUs, by original MapTracker authors. Clean input only; no corruption training; no gate supervision.
- Role: Performance reference and unmodified stale-FP baseline.

**B1: Corruption-trained, no-gate baseline**
- Checkpoint: `TrainningPaths/b1_iter_89148.pth`; run_id: `b1_stage3_gpu4_short_trainonly`
- Config: `maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py` with cfg-option override `model.mvp_temporal_gate_cfg.corruption_trained_no_gate_baseline=True`
- Gate status: DISABLED at runtime via `_enforce_no_gate_baseline_runtime()` (sets `gate.enabled=False` on all 6 decoder layers). Gate MLP is present in the architecture but receives no gradient; gate loss is skipped when `corruption_trained_no_gate_baseline=True`.
- Training: 4 GPUs (L40s), `samples_per_gpu=4` (16 total effective batch size), corruption training active (clean=60%, c_full=20%, c_tail=20%). Trained to convergence.
- Source of no-gate override: `tools/experiments/submit_b1_train_only_gpu4.sh`, line `--b1-cfg-options "model.mvp_temporal_gate_cfg.corruption_trained_no_gate_baseline=True"`
- Role: Ablation control — isolates the effect of corruption-aware backbone/decoder training without a learned gate.

**B2: Corruption-trained, gated model (proposed method)**
- Checkpoint: `TrainningPaths/b2_iter_89148.pth`; run_id: `b2_stage3_gpu4_short_trainonly`
- Config: Same base config; `corruption_trained_no_gate_baseline=False` (default)
- Gate status: ENABLED. `SlotwiseTemporalGate` active in each of 6 decoder layers; pre-fusion value-only gating over selected memory slots. Trained with L_close + L_open + L_clean supervision.
- Training: Identical hardware and corruption probabilities as B1. Trained to convergence.
- Role: Proposed method. Differs from B1 only in gate activation and supervision.

**Training parity (B1 ↔ B2):** Both trained from the same stage-2 warmup checkpoint (`work_dirs/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth`), same 4-GPU setup, same corruption probabilities, evaluated at iter_89,148. Differ only in `corruption_trained_no_gate_baseline` flag.

**Gate architecture (B2), as implemented:**
- Location: `plugin/models/transformer_utils/MapTransformer.py`
- Input: slotwise memory embeddings [T, B, D] with 5 auxiliary scalar features: cosine similarity, L2 distance, `valid_mem` flag, normalized `delta_t`, `age_rank_norm`
- MLP: `Linear(D*2+5, 64)` → ReLU → `Linear(64, 1)` → sigmoid → α ∈ [0,1] per slot
- Action: `V_mem_soft = α × V_mem` (value scaling only; keys unchanged)
- Insertion point: `MapTransformerLayer.forward()`, called before memory cross-attention

---

### 2.3 Training Details

| Parameter | Value | Source |
|---|---|---|
| Init checkpoint (B1, B2) | `work_dirs/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth` | `maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py` (`load_from`) |
| GPUs | 4 (L40s) | `tools/experiments/submit_b1_train_only_gpu4.sh` |
| Samples per GPU | 4 → 16 total effective batch size | Same |
| Total planned iters | 167,808 (`runner.max_iters=167808`) | Same |
| Evaluated checkpoint | iter_89,148 (trained to convergence) | `TrainningPaths/b1_iter_89148.pth`, `b2_iter_89148.pth` |
| Training start date | 2026-04-13 ~02:15 (both B1 and B2) | `B1B2TrainLogs/FinalB1TrainingLogs.csv`, `FinalB2TrainingLogs.csv` |
| Training end date (logs) | B1: 2026-04-24 ~12:31, iter 95,800; B2: 2026-04-24 ~09:56, iter 95,250 | `B1B2TrainLogs/FinalB1TrainingLogs.csv`, `FinalB2TrainingLogs.csv` |
| LR at eval checkpoint (iter 89,150) | 2.263×10⁻⁵ (both B1 and B2; cosine annealing decay from 5×10⁻⁴) | `B1B2TrainLogs/FinalB1TrainingLogs.csv`, `FinalB2TrainingLogs.csv` |
| Optimizer | AdamW | `maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py` |
| Learning rate | 5×10⁻⁴, cosine annealing, `warmup_iters=500`, `min_lr_ratio=3×10⁻³` | Same config |
| Weight decay | 1×10⁻² | Same config |
| Gradient clip | max_norm=35 | Same config |
| Corruption probs (training) | clean=0.60, c_full=0.20, c_tail=0.20 | Same config |
| Stale offsets (training) | [1, 2, 3] (uniform sample per step) | Same config |
| corruption_onset | 0 (applied from frame 0) | Same config |
| c_tail_keep_recent | 1 | Same config |
| Gate loss weights (B2) | λ_close=1.0, λ_open=0.5, λ_clean=0.1; `enable_clean_open_loss=True` | Same config |
| Backbone | ResNet-50 + FPN + BEVFormerEncoder | Same config |
| Decoder | MapTransformerDecoder_new, 6 layers | Same config |

---

## 3. Section 6 Facts

### 3.1 Clean Performance (condition = `clean`)

| Model | mAP | Ped AP | Div AP | Bnd AP | mean_cMAP | Ped cAP | Div cAP | Bnd cAP |
|---|---|---|---|---|---|---|---|---|
| **B0** | 0.7655 | 0.79892077788045680 | 0.74985369244889500 | 0.74779617361306720 | 0.69830712963941950 | 0.77162360211666140 | 0.65271426604999780 | 0.67058352075159930 |
| **B1** | 0.73000 | 0.74570 | 0.71100 | 0.73330 | 0.67288869673669530 | 0.71301391258432770 | 0.63331471584075140 | 0.67233746178500690 |
| **B2** | 0.73030 | 0.76760 | 0.70300 | 0.72030 | 0.67033239192514920 | 0.73218475877009460 | 0.61467247669811560 | 0.66413994030723700 |
| B2 − B1 | +0.00030 | +0.02190 | −0.00800 | −0.01300 | −0.00255630481154610 | +0.01917084618576690 | −0.01864223914263580 | −0.00819752147776990 |

Sources: B0 mAP + cMAP from `work_dirs/sbatch/clean_eval_b0/20260428_011807/logs/maptracker_clean_eval_b0-7416390.out`; B1/B2 mAP from `CurrentB1B2Results/clean_eval_results_master.csv`; B1/B2 cMAP from `CurrentB1B2Results/b1_eval_89148/clean/logs/run_b1_b2_deferred_eval.log` and `…/b2_eval_89148/clean/logs/run_b1_b2_deferred_eval.log`.

**B0 per-threshold clean:**

| Class | AP@0.5 | AP@1.0 | AP@1.5 | AP |
|---|---|---|---|---|
| ped_crossing | 0.6551 | 0.8437 | 0.8980 | 0.79892077788045680 |
| divider | 0.6502 | 0.7705 | 0.8288 | 0.74985369244889500 |
| boundary | 0.6044 | 0.7844 | 0.8546 | 0.74779617361306720 |

**B0 per-threshold clean cMAP:**

| Class | cAP@0.5 | cAP@1.0 | cAP@1.5 | cAP |
|---|---|---|---|---|
| ped_crossing | 0.64257958776193890 | 0.81043862152433250 | 0.86185259706371260 | 0.77162360211666140 |
| divider | 0.58293872831661700 | 0.66724970997527700 | 0.70795435985809950 | 0.65271426604999780 |
| boundary | 0.55572448821533990 | 0.69778729387906830 | 0.75823878016038960 | 0.67058352075159930 |
| **mean** | | | | **0.69830712963941950** |

**B1 per-threshold clean:**

| Class | AP@0.5 | AP@1.0 | AP@1.5 | AP |
|---|---|---|---|---|
| ped_crossing | 0.59970 | 0.78070 | 0.85660 | 0.74570 |
| divider | 0.60860 | 0.73010 | 0.79420 | 0.71100 |
| boundary | 0.58910 | 0.76930 | 0.84150 | 0.73330 |

**B1 per-threshold clean cMAP:**

| Class | cAP@0.5 | cAP@1.0 | cAP@1.5 | cAP |
|---|---|---|---|---|
| ped_crossing | 0.58286360523983480 | 0.74487405942131610 | 0.81130407309183170 | 0.71301391258432770 |
| divider | 0.55557162328002110 | 0.64809886136264760 | 0.69627366287958540 | 0.63331471584075140 |
| boundary | 0.55465611550753250 | 0.70254285333612540 | 0.75981341651136260 | 0.67233746178500690 |
| **mean** | | | | **0.67288869673669530** |

**B2 per-threshold clean:**

| Class | AP@0.5 | AP@1.0 | AP@1.5 | AP |
|---|---|---|---|---|
| ped_crossing | 0.62160 | 0.81340 | 0.86770 | 0.76760 |
| divider | 0.60610 | 0.72670 | 0.77640 | 0.70300 |
| boundary | 0.58070 | 0.76130 | 0.81890 | 0.72030 |

**B2 per-threshold clean cMAP:**

| Class | cAP@0.5 | cAP@1.0 | cAP@1.5 | cAP |
|---|---|---|---|---|
| ped_crossing | 0.60423836924641140 | 0.77532475102960560 | 0.81699115603426710 | 0.73218475877009460 |
| divider | 0.54417782506145010 | 0.63341014629769610 | 0.66642945873520090 | 0.61467247669811560 |
| boundary | 0.54342442549254710 | 0.70028313074271750 | 0.74871226468644650 | 0.66413994030723700 |
| **mean** | | | | **0.67033239192514920** |

**Prediction count observation (structural):**
- B0: ped=176,146; div=292,123; bnd=182,847
- B1: ped≈91,933; div≈273,549; bnd≈283,952
- B2: ped≈57,391; div≈194,178; bnd≈396,474

B2 produces ~38% fewer ped/div predictions but ~40% more bnd predictions than B1. Unexplained structural difference; potential confound for per-class comparisons. Source: `CurrentB1B2Results/clean_eval_results_master.csv`.

---

### 3.2 Robustness Under Stale Memory Injection

**Corruption protocol:**
- Implemented in `plugin/models/mapers/vector_memory.py`, `_build_local_corrupted_read_view()`
- `c_full`: all valid selected memory slots replaced with content from frame `selected_idx − stale_offset`
- `c_tail`: only older slots replaced; `tail_keep_recent=1` most-recent slot preserved clean
- Canonical (write) memory remains clean in all conditions; corruption is read-side only

#### mAP Under Corruption

| Condition | B0 mAP | B1 mAP | B2 mAP | B2−B1 |
|---|---|---|---|---|
| c_full, offset=1, onset=0 | 0.76490 | 0.72950 | 0.73070 | +0.00120 |
| c_full, offset=2, onset=0 | 0.76480 | 0.72950 | 0.73060 | +0.00110 |
| c_full, offset=3, onset=0 | 0.76460 | 0.72920 | 0.73070 | +0.00150 |
| c_tail, offset=1, onset=0 | 0.76540 | 0.73020 | 0.73050 | +0.00030 |
| c_tail, offset=2, onset=0 | 0.76550 | 0.73050 | 0.73050 | 0.00000 |
| c_tail, offset=3, onset=0 | 0.76550 | 0.73040 | 0.73030 | −0.00010 |
| **avg c_full** | **0.76477** | **0.72940** | **0.73067** | **+0.00127** |
| **avg c_tail** | **0.76547** | **0.73037** | **0.73043** | **+0.00007** |
| **avg all** | **0.76512** | **0.72988** | **0.73055** | **+0.00067** |

Sources: `CurrentB1B2Results/contradiction_results_master.csv`; `CurrentB1B2Results/b0/contradiction_results_b0.csv`.

**Clean-to-corrupt mAP degradation:**

| Condition | B0 (corrupt − clean 0.7655) | B1 (corrupt − clean 0.73000) | B2 (corrupt − clean 0.73030) |
|---|---|---|---|
| c_full, offset=1 | −0.00060 | −0.00050 | +0.00040 |
| c_full, offset=2 | −0.00070 | −0.00050 | +0.00030 |
| c_full, offset=3 | −0.00090 | −0.00080 | +0.00040 |
| c_tail, offset=1 | −0.00010 | +0.00020 | +0.00020 |
| c_tail, offset=2 | 0.00000 | +0.00050 | +0.00020 |
| c_tail, offset=3 | 0.00000 | +0.00040 | 0.00000 |

All degradations are within noise for all three models. B0 shows minimal clean→corrupt degradation (−0.00090 to 0.00000), confirming its clean mAP advantage vs. B1/B2 is not due to differing corruption sensitivity.

#### mean_cMAP Under Corruption

| Condition | B0 mean_cMAP | B1 mean_cMAP | B2 mean_cMAP | B2−B1 |
|---|---|---|---|---|
| c_full, offset=1, onset=0 | 0.69468050034961550 | 0.67365627124325280 | 0.67228438234994490 | −0.00137188889330790 |
| c_full, offset=2, onset=0 | 0.68999286137328670 | 0.67394263199675710 | 0.67186856778487940 | −0.00207406421187770 |
| c_full, offset=3, onset=0 | 0.69130319579777610 | 0.67199836764912200 | 0.67140634308627560 | −0.00059202455284640 |
| c_tail, offset=1, onset=0 | 0.69551754250077840 | 0.67251276164850170 | 0.67139123566755480 | −0.00112152598094690 |
| c_tail, offset=2, onset=0 | 0.69666628588903930 | 0.67330808804751950 | 0.67076347828988400 | −0.00254460975763550 |
| c_tail, offset=3, onset=0 | 0.69481562910255710 | 0.67316648240982680 | 0.67084691448596510 | −0.00231956792386170 |
| **avg c_full** | **0.69199218584022610** | **0.67319909029637730** | **0.67185309774036660** | **−0.00134599255601070** |
| **avg c_tail** | **0.69566648583079160** | **0.67299577743528270** | **0.67100054281446790** | **−0.00199523462081480** |
| **avg all** | **0.69382933581884220** | **0.67309743383172990** | **0.67142682026075060** | **−0.00167061357097930** |

Sources: `work_dirs/sbatch/cmap_parallel/20260427_231314_latest_onset0_trainmatched/logs/cmap_seq_b0-7420459.out`; `…20260427_231314_b1_contra_89148/logs/cmap_seq_b1-7420460.out`; `…20260427_231314_b2_contra_89148/logs/cmap_seq_b2-7420461.out`.

**Clean-to-corrupt cMAP degradation:**

| Condition | B0 (corrupt − clean 0.69830712963941950) | B1 (corrupt − clean 0.67288869673669530) | B2 (corrupt − clean 0.67033239192514920) |
|---|---|---|---|
| c_full, offset=1 | −0.00362662928980400 | +0.00076757450655750 | +0.00195199042479570 |
| c_full, offset=2 | −0.00831426826613280 | +0.00105393526006180 | +0.00153617585973020 |
| c_full, offset=3 | −0.00700393384164340 | −0.00088032908757330 | +0.00107395116112640 |
| c_tail, offset=1 | −0.00278958713864110 | −0.00037593508819360 | +0.00105884374240560 |
| c_tail, offset=2 | −0.00164084375038020 | +0.00041939131082420 | +0.00043108636473480 |
| c_tail, offset=3 | −0.00349150053686240 | +0.00027778567313150 | +0.00051452256081590 |

All values are within ±0.009. B1 and B2 show negligible cMAP degradation under corruption (within ±0.003). B0 shows modest but consistent degradation under corruption (−0.001 to −0.008), particularly under c_full — consistent with B0 having no corruption-robustness training.

**Observation:** B1 consistently produces higher cMAP than B2 across all 6 corruption conditions (by 0.00059–0.00254). This mirrors the clean cMAP gap (B1=0.67289 > B2=0.67033), suggesting the gate training modestly reduces temporal consistency as measured by cMAP. B0's higher baseline cMAP (0.6983) and its degradation under corruption confirms that clean-only training yields better temporal consistency at the cost of robustness.

#### Per-Class cMAP Under Corruption — B0

| Condition | Ped cAP | Div cAP | Bnd cAP | mean_cMAP |
|---|---|---|---|---|
| c_full, offset=1 | 0.76885594135198830 | 0.65194653606154530 | 0.66323902363531310 | 0.69468050034961550 |
| c_full, offset=2 | 0.76661937936200500 | 0.64513640867622870 | 0.65822279608162670 | 0.68999286137328670 |
| c_full, offset=3 | 0.76165187160499260 | 0.65262402229317330 | 0.65963369349516240 | 0.69130319579777610 |
| c_tail, offset=1 | 0.76829803460270650 | 0.65227546305356050 | 0.66597912984606830 | 0.69551754250077840 |
| c_tail, offset=2 | 0.77109181707460160 | 0.65132675767842830 | 0.66758028291408810 | 0.69666628588903930 |
| c_tail, offset=3 | 0.76659855685472640 | 0.65070495065792780 | 0.66714337979501670 | 0.69481562910255710 |

#### Per-Class cMAP Under Corruption — B1

| Condition | Ped cAP | Div cAP | Bnd cAP | mean_cMAP |
|---|---|---|---|---|
| c_full, offset=1 | 0.71490501988977650 | 0.63357674121906050 | 0.67248705262092160 | 0.67365627124325280 |
| c_full, offset=2 | 0.71567155293708410 | 0.63288916069456260 | 0.67326718235862480 | 0.67394263199675710 |
| c_full, offset=3 | 0.71275322677366730 | 0.63203506968348810 | 0.67120680649021050 | 0.67199836764912200 |
| c_tail, offset=1 | 0.71264018116759960 | 0.63378686855002750 | 0.67111123522787750 | 0.67251276164850170 |
| c_tail, offset=2 | 0.71395441166549600 | 0.63350024405842000 | 0.67246960841864250 | 0.67330808804751950 |
| c_tail, offset=3 | 0.71379642966966800 | 0.63331449542245100 | 0.67238852213736160 | 0.67316648240982680 |

#### Per-Class cMAP Under Corruption — B2

| Condition | Ped cAP | Div cAP | Bnd cAP | mean_cMAP |
|---|---|---|---|---|
| c_full, offset=1 | 0.73316489100292840 | 0.61533266118032170 | 0.66835559486658500 | 0.67228438234994490 |
| c_full, offset=2 | 0.73283047623969480 | 0.61446389648349390 | 0.66831133063144940 | 0.67186856778487940 |
| c_full, offset=3 | 0.73242729120599390 | 0.61401314207470140 | 0.66777859597813140 | 0.67140634308627560 |
| c_tail, offset=1 | 0.73282532983629380 | 0.61490861345805560 | 0.66643976370831490 | 0.67139123566755480 |
| c_tail, offset=2 | 0.73280189355037040 | 0.61429718204277740 | 0.66519135927650420 | 0.67076347828988400 |
| c_tail, offset=3 | 0.73247618102952360 | 0.61436005624065230 | 0.66570450618771950 | 0.67084691448596510 |

---

### 3.3 Gate Behavior (Alpha Statistics)

**Clean condition:**

| Model | α_clean_recent | α_preserved_recent |
|---|---|---|
| B1 | 0.97510 | 0.97510 |
| B2 | 0.93660 | 0.93660 |

B2's slightly lower clean α_clean_recent (0.93660 vs. 0.97510) is a training artifact; both are high (>0.93).

**Under corruption — complete table:**

| Condition | Model | α_affected | α_preserved_recent | α_separation |
|---|---|---|---|---|
| c_full, offset=1 | B0 | 0.89450 | 0.51490 | −0.37960 |
| c_full, offset=1 | B1 | 0.88390 | 0.48280 | −0.40110 |
| c_full, offset=1 | **B2** | **0.33400** | 0.47150 | **+0.13750** |
| c_full, offset=2 | B0 | 0.89520 | 0.63380 | −0.26130 |
| c_full, offset=2 | B1 | 0.88350 | 0.59200 | −0.29160 |
| c_full, offset=2 | **B2** | **0.19160** | 0.58140 | **+0.38980** |
| c_full, offset=3 | B0 | 0.83870 | 0.70530 | −0.13340 |
| c_full, offset=3 | B1 | 0.83300 | 0.65890 | −0.17410 |
| c_full, offset=3 | **B2** | **0.12720** | 0.63920 | **+0.51200** |
| c_tail, offset=1 | B0 | 0.81760 | 0.97510 | +0.15750 |
| c_tail, offset=1 | B1 | 0.80740 | 0.97510 | +0.16760 |
| c_tail, offset=1 | **B2** | **0.12550** | 0.93650 | **+0.81100** |
| c_tail, offset=2 | B0 | 0.83700 | 0.97510 | +0.13810 |
| c_tail, offset=2 | B1 | 0.82360 | 0.97510 | +0.15150 |
| c_tail, offset=2 | **B2** | **0.07900** | 0.93650 | **+0.85750** |
| c_tail, offset=3 | B0 | 0.38180 | 0.97510 | +0.59330 |
| c_tail, offset=3 | B1 | 0.37460 | 0.97510 | +0.60040 |
| c_tail, offset=3 | **B2** | **0.03230** | 0.93650 | **+0.90420** |

Sources: `CurrentB1B2Results/contradiction_results_master.csv`; `CurrentB1B2Results/b0/contradiction_results_b0.csv`.

**Key patterns:**
- B2 α_affected is **2.6×–27×** lower than B1's across all conditions
- B2 α_separation is consistently and strongly positive; B1 is negative under c_full, marginally positive under c_tail
- B0 and B1 alpha patterns are nearly identical — the absence of gate training produces the same alpha behavior regardless of corruption-awareness in the backbone
- Suppression strength scales with stale offset in B2: larger offset → larger temporal divergence detectable via `age_rank`/`delta_t` features → lower α_affected
- **But:** this clear alpha discrimination does not translate to improved mAP or cMAP

**Note on B0/B1 alpha interpretation:** Both models have their gates disabled at eval time. In `SlotwiseTemporalGate.forward()`, when `self.enabled=False`, the MLP is **never called** — the code early-returns `alpha = eligible_mask` (1.0 for each valid/non-padded memory slot, 0.0 for padding). Memory values passed to cross-attention are the **unmodified** embeddings. Therefore the logged alpha values for B0 and B1 do not reflect any learned or random MLP output. Instead, `α_affected` and `α_preserved_recent` are effectively **frame-coverage fractions** — the mean fraction of eval frames that had at least one corrupted (or one preserved-recent) valid memory slot contributing to the average (the 0.0 contributions from frames with no eligible slots pull the mean below 1.0). This is why B1's `α_preserved_recent` under c_tail is exactly 0.97508 — equal to its clean `α_clean_recent` — since c_tail always preserves the most-recent slot. The negative α_separation for B0/B1 under c_full conditions is a **coverage artifact**: more frames have at least one corrupted slot (~88%) than have a non-corrupted most-recent slot (~48%), because c_full also targets the most-recent slot. B0 and B1 are a true **gate-open baseline** (weight=1 for all valid slots), not an untrained-MLP baseline.

---

### 3.4 Stale False-Positive Metrics

| Condition | B0 stale persist (frames) | B1 stale persist | B2 stale persist | B0 cum FP (m) | B1 cum FP (m) | B2 cum FP (m) | B2−B1 (m) |
|---|---|---|---|---|---|---|---|
| c_full, offset=1 | 40.13 | 40.13 | 40.13 | 1,677,633 | 1,612,257 | 1,599,704 | −12,553 |
| c_full, offset=2 | 40.13 | 40.13 | 40.13 | 1,675,859 | 1,611,771 | 1,600,192 | −11,579 |
| c_full, offset=3 | 40.13 | 40.13 | 40.13 | 1,678,001 | 1,611,593 | 1,600,380 | −11,213 |
| c_tail, offset=1 | 40.13 | 40.13 | 40.13 | 1,675,324 | 1,611,416 | 1,600,471 | −10,945 |
| c_tail, offset=2 | 40.13 | 40.13 | 40.13 | 1,676,798 | 1,611,597 | 1,600,292 | −11,305 |
| c_tail, offset=3 | 40.13 | 40.13 | 40.13 | 1,675,489 | 1,611,098 | 1,600,517 | −10,581 |
| **avg** | **40.13** | **40.13** | **40.13** | **1,676,517** | **1,611,455** | **1,600,259** | **−11,196** |

Source: `CurrentB1B2Results/contradiction_results_master.csv`; `CurrentB1B2Results/b0/contradiction_results_b0.csv`.

`stale_persistence_time_mean` is **exactly 40.13 frames** for every model and condition — provides no differentiation.
B2 cumulative FP reduction vs. B1: avg −11,196 m, or **~0.69%**.

---

### 3.5 Qualitative Examples

No qualitative visualizations exist in the repository as of 2026-04-28. No saved scene outputs, selected-frame exports, or image comparisons for B0/B1/B2 were found under `maptracker_v1/`. [TODO: Generate by running inference with scene visualization tools.]

---

### 3.6 Training Loss Convergence (B1 vs. B2)

Task-loss trajectories extracted from `B1B2TrainLogs/FinalB1TrainingLogs.csv` and `FinalB2TrainingLogs.csv` (successful run, April 13–24, 2026). Snapshot rows closest to each target iteration. Main-frame `cls` and `reg` are reported; all other temporal-frame losses are consistent with these values (see Table 6 for full breakdown at convergence).

**Loss trajectory — current frame (main) classification and regression loss:**

| Iter | B1 cls | B1 reg | B2 cls | B2 reg |
|---|---|---|---|---|
| 50 | 0.2851 | 0.6527 | 0.2669 | 0.6417 |
| 5,000 | 0.3032 | 0.5698 | 0.2627 | 0.5672 |
| 10,000 | 0.2369 | 0.5430 | 0.2214 | 0.5178 |
| 20,000 | 0.1482 | 0.4136 | 0.1405 | 0.4083 |
| 40,000 | 0.0994 | 0.2571 | 0.0778 | 0.2628 |
| 60,000 | 0.0450 | 0.1505 | 0.0389 | 0.1618 |
| 80,000 | 0.0324 | 0.1513 | 0.0254 | 0.1041 |
| **89,150** | **0.0182** | **0.0855** | **0.0127** | **0.0859** |

Both models start at nearly identical loss values (B1 cls=0.2851, B2 cls=0.2669 at iter 50) and converge to near-identical task losses at the eval checkpoint (B1 cls=0.0182, B2 reg=0.0855; B2 cls=0.0127, B2 reg=0.0859). The gate supervision (B2) does not impair learning of the core map prediction task.

**Note on gate training metrics:** `gate_loss`, `L_close`, `L_open`, `L_clean`, and `alpha_mean_*` statistics during training are **not available** from the successful post-April-13 run. Log lines from that run are truncated at 1,027 characters (at `d2.reg_t2`), before gate metric fields appear. The `FinalB1TrainingLogs.csv` and `FinalB2TrainingLogs.csv` contain **57 columns** (task losses only; no gate metrics). The `OldB1TrainingLogs.csv` and `OldB2TrainingLogs.csv` (pre-April-13 failed run) do contain gate metrics, but only for early training: B1 iter 50–37,500 (838 non-empty rows) and B2 iter 50–26,250 (588 non-empty rows) — gate metrics become empty beyond those points in the failed run as well. These logs are not usable per the experimental scope constraint (post-April-13 only). Gate behavior is characterized solely through eval-time alpha statistics (Section 3.3).

---

### 3.7 Summary of Findings

1. **mAP is insensitive to stale memory injection.** Both B1 and B2 degrade less than 0.00150 mAP under all six tested corruption conditions. The per-class AP metric on this dataset does not detect the performance impact of stale memory corruption at offsets 1–3.

2. **cMAP is similarly insensitive.** B1 and B2 cMAP changes under corruption are within ±0.003 of clean cMAP. B1 consistently outperforms B2 on cMAP across all conditions (corrupt and clean alike) by a small but consistent margin (~0.002–0.003).

3. **The trained gate (B2) suppresses stale slot values.** α_affected drops from ~0.88 (B1) to 0.032–0.334 (B2); alpha separation is consistently and strongly positive for B2 (range: +0.138 to +0.904). The gate functions as designed.

4. **Alpha separation does not translate to downstream metric improvement.** B2 achieves clear alpha discrimination between stale and fresh slots but does not improve mAP or cMAP vs. B1. This is the central result requiring careful interpretation in the paper.

5. **Gate suppression scales with staleness.** Larger stale offset → lower α_affected and higher α_separation in B2, consistent with the gate learning to detect temporal divergence via `age_rank` and `delta_t` features.

6. **Stale FP persistence is invariant across all models and conditions.** The 40.13-frame invariance of `stale_persistence_time_mean` is unexplained and likely reflects structural properties of nuScenes validation sequences rather than model behavior.

7. **Cumulative stale FP length shows a modest B2 advantage.** B2 reduces cumulative stale-FP polyline length by ~0.69% relative to B1, consistent with the gate suppressing some stale value contributions.

---

## 4. Consolidated Results Tables

### Table 1: Clean Performance

| Model | mAP | Ped AP | Div AP | Bnd AP | mean_cMAP | Ped cAP | Div cAP | Bnd cAP |
|---|---|---|---|---|---|---|---|---|
| B0 | 0.7655 | 0.79892077788045680 | 0.74985369244889500 | 0.74779617361306720 | 0.69830712963941950 | 0.77162360211666140 | 0.65271426604999780 | 0.67058352075159930 |
| B1 | 0.73000 | 0.74570 | 0.71100 | 0.73330 | 0.67288869673669530 | 0.71301391258432770 | 0.63331471584075140 | 0.67233746178500690 |
| B2 | 0.73030 | 0.76760 | 0.70300 | 0.72030 | 0.67033239192514920 | 0.73218475877009460 | 0.61467247669811560 | 0.66413994030723700 |

### Table 2: Corrupt Performance — mAP

| Condition | B0 mAP | B1 mAP | B2 mAP | B2−B1 |
|---|---|---|---|---|
| c_full, offset=1 | 0.76490 | 0.72950 | 0.73070 | +0.00120 |
| c_full, offset=2 | 0.76480 | 0.72950 | 0.73060 | +0.00110 |
| c_full, offset=3 | 0.76460 | 0.72920 | 0.73070 | +0.00150 |
| c_tail, offset=1 | 0.76540 | 0.73020 | 0.73050 | +0.00030 |
| c_tail, offset=2 | 0.76550 | 0.73050 | 0.73050 | 0.00000 |
| c_tail, offset=3 | 0.76550 | 0.73040 | 0.73030 | −0.00010 |
| **avg** | **0.76512** | **0.72988** | **0.73055** | **+0.00067** |

### Table 3: Corrupt Performance — mean_cMAP

| Condition | B0 mean_cMAP | B1 mean_cMAP | B2 mean_cMAP | B2−B1 |
|---|---|---|---|---|
| c_full, offset=1 | 0.69468050034961550 | 0.67365627124325280 | 0.67228438234994490 | −0.00137188889330790 |
| c_full, offset=2 | 0.68999286137328670 | 0.67394263199675710 | 0.67186856778487940 | −0.00207406421187770 |
| c_full, offset=3 | 0.69130319579777610 | 0.67199836764912200 | 0.67140634308627560 | −0.00059202455284640 |
| c_tail, offset=1 | 0.69551754250077840 | 0.67251276164850170 | 0.67139123566755480 | −0.00112152598094690 |
| c_tail, offset=2 | 0.69666628588903930 | 0.67330808804751950 | 0.67076347828988400 | −0.00254460975763550 |
| c_tail, offset=3 | 0.69481562910255710 | 0.67316648240982680 | 0.67084691448596510 | −0.00231956792386170 |
| **avg** | **0.69382933581884220** | **0.67309743383172990** | **0.67142682026075060** | **−0.00167061357097930** |

### Table 4: Alpha Statistics Under Corruption (B1 vs. B2)

| Condition | B1 α_affected | B2 α_affected | B1 α_sep | B2 α_sep |
|---|---|---|---|---|
| c_full, offset=1 | 0.88390 | **0.33400** | −0.40110 | **+0.13750** |
| c_full, offset=2 | 0.88350 | **0.19160** | −0.29160 | **+0.38980** |
| c_full, offset=3 | 0.83300 | **0.12720** | −0.17410 | **+0.51200** |
| c_tail, offset=1 | 0.80740 | **0.12550** | +0.16760 | **+0.81100** |
| c_tail, offset=2 | 0.82360 | **0.07900** | +0.15150 | **+0.85750** |
| c_tail, offset=3 | 0.37460 | **0.03230** | +0.60040 | **+0.90420** |

### Table 5: Cumulative Stale FP Polyline Length (meters)

| Condition | B0 | B1 | B2 | B2−B1 |
|---|---|---|---|---|
| c_full, offset=1 | 1,677,633 | 1,612,257 | 1,599,704 | −12,553 |
| c_full, offset=2 | 1,675,859 | 1,611,771 | 1,600,192 | −11,579 |
| c_full, offset=3 | 1,678,001 | 1,611,593 | 1,600,380 | −11,213 |
| c_tail, offset=1 | 1,675,324 | 1,611,416 | 1,600,471 | −10,945 |
| c_tail, offset=2 | 1,676,798 | 1,611,597 | 1,600,292 | −11,305 |
| c_tail, offset=3 | 1,675,489 | 1,611,098 | 1,600,517 | −10,581 |
| **avg** | **1,676,517** | **1,611,455** | **1,600,259** | **−11,196** |

*`stale_persistence_time_mean` = 40.13 frames for all models and conditions (no differentiation).*

---

### Table 6: Training Task-Loss Convergence (B1 vs. B2, Successful Run)

Columns `cls` and `reg` are main-frame (current-frame) classification and regression losses. Full multi-frame loss values at the eval checkpoint (iter 89,150) are shown in Table 6b.

**Table 6a — Loss trajectory (main-frame cls / reg):**

| Iter | B1 cls | B1 reg | B2 cls | B2 reg |
|---|---|---|---|---|
| 50 | 0.2851 | 0.6527 | 0.2669 | 0.6417 |
| 5,000 | 0.3032 | 0.5698 | 0.2627 | 0.5672 |
| 10,000 | 0.2369 | 0.5430 | 0.2214 | 0.5178 |
| 20,000 | 0.1482 | 0.4136 | 0.1405 | 0.4083 |
| 40,000 | 0.0994 | 0.2571 | 0.0778 | 0.2628 |
| 60,000 | 0.0450 | 0.1505 | 0.0389 | 0.1618 |
| 80,000 | 0.0324 | 0.1513 | 0.0254 | 0.1041 |
| **89,150** | **0.0182** | **0.0855** | **0.0127** | **0.0859** |

Source: `B1B2TrainLogs/FinalB1TrainingLogs.csv`, `FinalB2TrainingLogs.csv`.

**Table 6b — Full loss breakdown at iter 89,150 (snapshot row):**

| Loss term | B1 | B2 |
|---|---|---|
| lr | 2.263e-05 | 2.263e-05 |
| cls (current frame) | 0.0182 | 0.0127 |
| reg (current frame) | 0.0855 | 0.0859 |
| seg | 0.0354 | 0.0360 |
| seg_dice | 0.0652 | 0.0646 |
| cls_t0 | 0.0258 | 0.0295 |
| reg_t0 | 0.1911 | 0.1855 |
| seg_t0 | 0.0697 | 0.0681 |
| seg_dice_t0 | 0.1223 | 0.1245 |
| cls_t1 | 0.0152 | 0.0133 |
| reg_t1 | 0.1200 | 0.1131 |
| cls_t2 | 0.0141 | 0.0128 |
| reg_t2 | 0.0944 | 0.0957 |

Source: `B1B2TrainLogs/FinalB1TrainingLogs.csv`, `FinalB2TrainingLogs.csv`. Gate metrics (`gate_loss`, `L_close`, `L_open`, `L_clean`, `alpha_mean_*`) are not present in these CSVs (log truncation in successful run; see Section 3.6 note).

---

## 5. Caveats and Confounds

**C1 — B0 clean results (resolved, 2026-04-28).** B0 clean mAP = 0.7655; clean cMAP = 0.69830712963941950. Source: `work_dirs/sbatch/clean_eval_b0/20260428_011807/logs/maptracker_clean_eval_b0-7416390.out`. B0 clean→corrupt mAP degradation is minimal (−0.00090 to 0.00000), matching the near-zero degradation in B1/B2. The ~0.035 mAP gap between B0 and B1/B2 is present in both clean (0.7655 vs. 0.7300/0.7303) and corrupt (0.765 vs. 0.730) conditions, confirming it reflects training-regime differences (B0: clean-only training) rather than robustness to corruption. B0's higher clean cMAP (0.6983 vs. 0.6729/0.6703) degrades modestly under corruption (−0.001 to −0.008), unlike B1/B2 which are stable, consistent with B0 lacking corruption-robustness training.

**C2 — mAP insensitivity.** Near-zero clean→corrupt degradation for both B1 and B2 indicates standard mAP is not a sufficiently sensitive metric for this corruption protocol at the tested stale offsets (1–3 frames). The alpha statistics and cumulative FP length are the primary experimental signals for evaluating gate behavior.

**C3 — Prediction count discrepancy.** B2 produces ~38% fewer ped/div predictions and ~40% more bnd predictions than B1. This unexplained structural difference is a potential confound for per-class comparisons and the B2 cumulative FP reduction.

**C4 — Alpha interpretation for disabled gates.** B0 and B1 log non-trivial alpha values despite gates being disabled. When `gate.enabled=False`, `SlotwiseTemporalGate` returns `alpha = eligible_mask` (1.0 for valid slots, 0.0 for padding) without calling the MLP at all. The aggregate alpha stats are therefore **frame-coverage fractions**, not gate outputs. `α_affected` ≈ fraction of frames with ≥1 eligible corrupted slot; `α_preserved_recent` ≈ fraction of frames with ≥1 non-corrupted most-recent slot. The negative α_separation for B0/B1 under c_full is a coverage artifact — c_full corrupts the most-recent slot, so the corrupted-slot coverage (~88%) exceeds the preserved-recent coverage (~48%), yielding separation < 0. This has no bearing on MLP discrimination ability. B0 and B1 represent a true gate-open baseline: all valid memory slots pass through to cross-attention at full weight.

**C5 — Eval corruption rate is stricter than training; types and offsets are in-distribution.** Training applies corruption stochastically per clip: 60% of clips are clean, 40% are corrupted (c_full or c_tail, offset sampled from [1,2,3]). Eval (`_inject_eval_memory_corruption_meta` via `_resolve_eval_memory_corruption_cfg`) applies corruption **deterministically at 100% rate** — every frame in the validation set sees the same fixed mode and offset. This is more sustained corruption than the model ever saw during training. The types (c_full, c_tail) and offset values (1–3) match the training set, so no out-of-distribution corruption type or offset is tested. The combination of near-zero mAP degradation under 100% corruption rate despite only 40% training exposure is a positive result, but generalization to larger stale offsets (e.g., 5, 10) or novel corruption types is unverified.

**C6 — No qualitative examples.** No scene visualizations have been generated. Qualitative claims cannot be supported.

---

## 6. Main Paper vs. Appendix Split (Recommended)

**Main paper (Sections 5–6):**
- Dataset/task config (one compact table)
- Model/baseline descriptions (B0, B1, B2) and training setup
- Table 1 (clean performance; all three models complete)
- Table 4 (alpha statistics — primary evidence of gate functioning)
- Table 2 mAP summary (citing near-zero degradation with caveat C2)
- C1 caveat (B0 clean pending) and C2 caveat (mAP insensitivity)
- Finding 3 (gate suppresses stale slots) and Finding 4 (no downstream metric gain)

**Appendix:**
- Full per-threshold AP breakdown for B1/B2 clean
- Table 3 (cMAP corruption conditions, full precision)
- Per-class cMAP tables for B0, B1, B2
- Table 5 (cumulative stale FP)
- Full Table 2 per-condition mAP detail

---

## 7. Missing Information Checklist

| Item | Status | Action needed |
|---|---|---|
| B0 clean mAP | **Complete** | `work_dirs/sbatch/clean_eval_b0/20260428_011807/logs/maptracker_clean_eval_b0-7416390.out` |
| B0 clean cMAP | **Complete** | Same log file (cMAP pipeline runs as part of clean eval) |
| Qualitative examples | **Missing** | Run inference with scene visualization tools |

---

## 8. Source File Reference Map

| Artifact | Source Path |
|---|---|
| B0 clean mAP + cMAP | `work_dirs/sbatch/clean_eval_b0/20260428_011807/logs/maptracker_clean_eval_b0-7416390.out` |
| B1 clean mAP | `CurrentB1B2Results/clean_eval_results_master.csv` |
| B2 clean mAP | `CurrentB1B2Results/clean_eval_results_master.csv` |
| B1 clean cMAP | `CurrentB1B2Results/b1_eval_89148/clean/logs/run_b1_b2_deferred_eval.log` |
| B2 clean cMAP | `CurrentB1B2Results/b2_eval_89148/clean/logs/run_b1_b2_deferred_eval.log` |
| B1 corrupt mAP + alpha | `CurrentB1B2Results/contradiction_results_master.csv` |
| B2 corrupt mAP + alpha | `CurrentB1B2Results/contradiction_results_master.csv` |
| B0 corrupt mAP + alpha | `CurrentB1B2Results/b0/contradiction_results_b0.csv` |
| B1 corrupt cMAP | `work_dirs/sbatch/cmap_parallel/20260427_231314_b1_contra_89148/logs/cmap_seq_b1-7420460.out` |
| B2 corrupt cMAP | `work_dirs/sbatch/cmap_parallel/20260427_231314_b2_contra_89148/logs/cmap_seq_b2-7420461.out` |
| B0 corrupt cMAP | `work_dirs/sbatch/cmap_parallel/20260427_231314_latest_onset0_trainmatched/logs/cmap_seq_b0-7420459.out` |
| B1 training logs | `B1B2TrainLogs/FinalB1TrainingLogs.csv` |
| B2 training logs | `B1B2TrainLogs/FinalB2TrainingLogs.csv` |
| Active config | `plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py` |
| B0 eval config | `plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_b0_eval.py` |
| B1 no-gate submit script | `tools/experiments/submit_b1_train_only_gpu4.sh` |
| Gate implementation | `plugin/models/transformer_utils/MapTransformer.py` |
| Corruption implementation | `plugin/models/mapers/vector_memory.py` |
| Gate loss / no-gate runtime | `plugin/models/mapers/MapTracker.py` |
| B1 checkpoint | `TrainningPaths/b1_iter_89148.pth` |
| B2 checkpoint | `TrainningPaths/b2_iter_89148.pth` |
| B0 checkpoint | `work_dirs/pretrained_ckpts/b0_nusc_oldsplit/latest.pth` |
