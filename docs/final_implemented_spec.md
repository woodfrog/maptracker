# MapTracker MVP — Implementation-Corrected Specification

*Pre-Fusion Slotwise Temporal Memory Gating for Stale-Memory Suppression*

This document is a **copy-derived, tightly edited correction** of `docs/mvp_spec.md` to reflect what is actually implemented in this repository.

---

## 1. Source basis and intent

This specification preserves the MVP structure and intent, but updates claims and contracts to match implemented code paths.

Authoritative implementation files:
- `plugin/models/transformer_utils/MapTransformer.py`
- `plugin/models/mapers/vector_memory.py`
- `plugin/models/mapers/MapTracker.py`
- `tools/test.py`
- `tools/tracking/run_contradiction_suite.py`
- `tools/tracking/contradiction_metrics.py`

---

## 2. Corrected research question

> Can the implemented pre-fusion temporal gate in the decoder memory branch suppress stale historical contribution under synthetic corruption controls, while maintaining usable clean-performance behavior?

This is evaluated primarily through implemented logging/metrics pathways and contradiction-suite tooling.

---

## 3. Scope: implemented keep/exclude state

### Implemented and active
- Pre-fusion value gating before memory-attention value consumption.
- Slotwise alpha gating over selected memory slots.
- Corruption modes: `clean`, `c_full`, `c_tail`.
- Read-path-only corrupted view with canonical memory isolation.
- Gate supervision losses: `L_close`, `L_open`, optional `L_clean`.
- No-gate baseline runtime disable mechanism (`corruption_trained_no_gate_baseline`).

### Implemented with corrected caveats
- Mask semantics are enforced effectively through tracked-index validity + query padding + valid memory, but explicit propagated/new/pad first-class mask contract is only partially surfaced.
- Training-stage intent (warmup/co-adaptation) is represented in config fields; strict parity/execution manifests are partially procedural.

### Not implemented in active MVP path
- BEV reliability branch (`c_ent`, `c_coh`, `c_rel`).
- Additive attention-logit bias.
- Bilinear disagreement gate block.
- Geometry-localized routing (`geom_fail_mask`, `prev_query_L_trans`).
- Translation/rotation corruption-mode scheduler (corruption modes remain `clean/c_full/c_tail`).

---

## 4. MapTracker context corrections

- Gate is wired on tracked-query memory branch indices (`valid_track_idx`).
- Gate operates on selected memory-bank entries, not full unselected history.
- Contradiction-focused tooling is present (`run_contradiction_suite.py`, `contradiction_metrics.py`).

---

## 5. Architecture and insertion point (corrected)

Implemented decoder memory-branch order:
1. Load selected memory embeddings and metadata for valid tracked queries.
2. Build effective eligibility (`valid_mem` + non-padded tracked queries).
3. Run temporal gate to produce `alpha` and `mem_values`.
4. Run memory attention with original memory embeddings as key-like input and gated values as value input.
5. Merge memory output with BEV-updated query path.

**Important implementation constraint:** gate forward path is packed with `q_len == 1` assertion in this branch.

---

## 6. Tensor contract (implementation-corrected)

The implemented gate interface is slotwise embedding/value oriented, not explicit per-head `K_mem`/`V_mem` tensors at gate entry.

### Persisted memory metadata (implemented)
- `batch_valid_mem_dict`
- `batch_delta_t_int_dict`
- `batch_age_rank_norm_dict`
- `batch_slot_corrupt_mask_dict`
- `batch_slot_corrupt_eligible_dict`
- `batch_mem_embeds_dict` (corrupted read view)
- `batch_mem_clean_embeds_dict` (clean selected view)

### Output tensors used downstream
- `batch_alpha_soft_dict`
- `batch_v_mem_soft_dict`

---

## 7. Mask semantics (implementation-corrected)

Implemented effective policy:
- Only valid tracked query indices enter memory branch gating.
- Query padding mask suppresses memory use for padded tracked entries.
- Invalid memory slots are excluded by slot validity and key padding.
- Ineligible positions force alpha suppression via eligibility masking in gate forward.

Note: explicit propagated/new/pad tensor API is not uniformly exposed as in the idealized MVP contract.

---

## 8. Gate design (implementation-corrected)

`SlotwiseTemporalGate` implementation:
- MLP hidden dimension: 64.
- Features include normalized query/memory embeddings, cosine similarity, L2 difference, valid-memory scalar, normalized delta-t, age-rank scalar.
- Output alpha through sigmoid.
- Action mechanism: value scaling (`gated_values = mem_embeds * alpha`).

Corrected divergence from original MVP text:
- No explicit sinusoidal age PE block in gate input path.
- No bilinear disagreement term.

---

## 9. Corruption protocol (implemented)

Implemented in `_build_local_corrupted_read_view`:
- Modes: `clean`, `c_full`, `c_tail`.
- Missing stale source => slot remains unsupervised/ineligible.
- Corrupted read view is local-forward artifact.
- Canonical memory write state remains clean.
- Onset and stale-offset are injected through metadata.

---

## 10. Loss design (implemented)

In `_compute_gate_supervision`:
- `L_close`: BCE on affected corrupted slots toward 0.
- `L_open`: BCE on preserved recent slots toward 1.
- `L_clean`: optional low-weight clean recent supervision.
- Gate loss weights configurable via `mvp_temporal_gate_cfg.gate_loss_weights`.

Base MapTracker losses remain active; gate loss is add-on when enabled.

---

## 11. Code modification map (implemented)

- Transformer/gate wiring: `MapTransformer.py`.
- Memory corruption + slot labels + metadata dicts: `vector_memory.py`.
- Loss integration + logging + no-gate enforcement: `MapTracker.py`.
- Eval-time corruption overrides + artifact writes: `tools/test.py`.
- Contradiction suite matrix and metrics: `tools/tracking/*`.

---

## 12. Training and baseline support (implemented status)

- Stage metadata (`freeze_stage`, `unfreeze_stage`) and corruption controls are present in config families.
- No-gate baseline switch is implemented and runtime-enforced in model code.
- Strict parity proof (sampler/steps/checkpoint-rule equivalence artifacts) remains partially procedural and should be runbooked per experiment.

---

## 13. Evaluation and outputs (implemented)

Implemented controls in `tools/test.py`:
- `--memory-corruption-mode`
- `--memory-stale-offset`
- `--memory-c-tail-keep-recent`
- `--memory-corruption-onset`
- `--condition-tag`

Implemented artifacts:
- `submission_vector.json`
- `condition_meta.json`
- `alpha_stats.json`
- `contradiction_metrics.json`
- `contradiction_suite_summary.json`

---

## 14. Required logging and test status

Implemented training logs include:
- `L_close`, `L_open`, `L_clean`
- `alpha_mean_affected`
- `alpha_mean_preserved_recent`
- `alpha_mean_clean_recent`
- `affected_batch_fraction`
- `eligible_slot_fraction`
- `historical_path_strength_ratio_clean`

---

## 15. Divergence summary (implementation-corrected)

Tracked divergences include:
1. Packed-path gate constraint (`q_len == 1`).
2. Gate feature-set drift from original richer MVP proposal.
3. Implicit vs explicit propagated/new/pad mask taxonomy.
4. Slotwise embedding/value gate-interface contract vs explicit per-head K/V contract.
5. Runtime verification maturity gap in current environment.

Divergence disposition should be tracked in `divergence_register.md`.

