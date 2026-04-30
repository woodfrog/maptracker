# MapTracker MVP Implementation Plan

*Pre-Fusion Slotwise Temporal Memory Gating for Stale-Memory Suppression*

A self-contained implementation specification for Independent Work

Prepared as a synthesis of the original proposal, disagreement-gated overview, prior code guide, TechSpec_v3, the MapTracker paper, and the public MapTracker repository README.

**Status:** Recommended MVP specification replacing the first-build target in TechSpec_v3

**Executive summary.** This document defines the exact minimum viable version of the project: a pre-fusion, slotwise temporal memory gate inside the MapTracker VEC decoder that suppresses stale historical vector memory before historical attention. The MVP keeps the core scientific contribution and removes the highest-risk sources of engineering overhead: multi-head independent gates, the BEV reliability branch, additive logit bias, geometry-localized failure routing, translation/rotation corruption modes, and the natural inconsistency subset. The goal is to maximize the probability of finishing a clean, defendable IW with meaningful results rather than a larger but fragile system.

## 1. Source basis and intended use

This document is designed to be self-contained. It should be sufficient for implementation planning, coding, experiment setup, and future-model context loading without requiring a new model to infer missing details from older specs.

It synthesizes the following source materials into one executable plan:

- Original project proposal: contradiction-aware gating of priors against current perception, corruption training, and contradiction-focused evaluation.
- Disagreement-Gated Decoder Overview: suggest-then-verify framing and the earlier multi-head / bilinear design rationale.
- Code Implementation for Disagreement-Gated Decoder: initial MapTransformer.py and vector_memory.py modification path.
- TechSpec_v3: the most complete operational specification, including masks, corruption semantics, losses, ablations, and diagnostics.
- MapTracker paper: the host model, memory formulation, training stages, and consistency-aware benchmark context.
- MapTracker GitHub repository README: top-level repo structure, training/evaluation guides, and terminology alignment.

This plan deliberately does not try to preserve every v3 mechanism. It preserves the minimal set of elements needed to answer the true research question with a feasible first implementation.

## 2. Final research question and success criteria

The project should now be framed around a narrower, cleaner question:

> **Can a pre-fusion, slotwise temporal memory gate inside MapTracker suppress stale historical vector memory before historical attention, improving contradiction recovery under controlled stale-memory corruption without materially hurting clean mapping performance?**

This question is intentionally narrower than the original proposal. It no longer attempts to solve temporal and spatial prior gating at once, and it does not require a learned perception-reliability branch in the first build.

| Dimension | Success criterion for the MVP |
| --- | --- |
| Core mechanism | The gate lowers the contribution of stale historical slots before historical attention output is formed. |
| Contradiction recovery | Under synthetic stale-memory corruption, stale geometry disappears faster than in the corruption-trained no-gate baseline. |
| Clean performance | Clean mAP / C-mAP does not materially collapse relative to the standard MapTracker baseline and the corruption-trained no-gate baseline. |
| Selectivity | Under C-tail, the gate assigns lower alpha to corrupted old slots than to preserved recent slots. |
| Feasibility | The entire system can be trained, debugged, and evaluated within an IW timeline without requiring invasive BEV operator surgery. |

## 3. Final scope decision

The table below is the authoritative keep / defer / exclude decision set for the MVP.

| Category | Decision | Reason |
| --- | --- | --- |
| Pre-fusion gating before historical attention | Keep | This is the central architectural idea and cannot be tested with late residual fusion. |
| Slotwise temporal gating over memory slots | Keep | Needed to tell the selective-staleness story, especially under C-tail. |
| Scalar gate per query-slot shared across heads | Keep | Preserves slotwise selectivity while cutting implementation risk compared with per-head gating. |
| Explicit propagated / new / pad masks | Keep | Correctness requirement; prevents invalid history use and silent bugs. |
| Value scaling only | Keep | Stable and sufficient for the first pass. |
| Layer-specific gate per decoder layer | Keep | Low overhead and matches the decoder structure. |
| Read-path corruption with clean-write memory | Keep | Necessary for causal, interpretable corruption training. |
| C-full and C-tail corruption modes | Keep | Directly target stale-memory rejection and slotwise selectivity. |
| Direct slot-level gate supervision on synthetically labeled slots | Keep | Simple, aligned with the corruption mechanism, and feasible. |
| Corruption-trained no-gate baseline | Keep | Needed to separate the effect of gating from the effect of corruption-trained data. |
| Multi-head independent gating | Exclude from MVP | Adds tensor/mask complexity without being necessary to establish the main result. |
| Bilinear disagreement feature | Exclude from MVP | Not needed for the first causal story and makes attribution harder. |
| BEV reliability branch: c_ent, c_coh, c_rel | Exclude from MVP | Too much code churn and debugging burden for a first implementation. |
| Additive attention-logit bias | Exclude from MVP | A refinement, not a requirement; value scaling should be proved first. |
| geom_fail_mask and prev_query_L_trans routing | Exclude from MVP | Only needed for more complex corruption modes; too heavy for the first build. |
| Translation / rotation corruption modes | Exclude from MVP | Broaden the scope from stale memory to general temporal misalignment too early. |
| Natural inconsistency subset | Defer | Good later analysis, not a first milestone. |
| Held-out stale offsets | Defer | Useful second-pass generalization check after the basic mechanism works. |
| Hard skip | Defer | Adds threshold behavior and extra failure modes. |
| Large ablation matrix and full diagnostics stack | Defer | Build only the measurements needed to finish a clean IW first. |

## 4. MapTracker context the implementation must respect

The MVP is not built in isolation. It lives inside MapTracker, which already treats vector HD mapping as a tracking problem with strided temporal memory fusion. That host structure matters for both coding and claims.

| Host-model fact | Implication for the MVP |
| --- | --- |
| MapTracker propagates vector queries across frames with PropMLP. | The gate should operate on propagated queries, not on brand-new queries with no temporal history. |
| MapTracker uses selected historical memory latents with a strided memory mechanism. | The gate should work over the selected slot set, not over the full raw history buffer. |
| MapTracker already has a multi-stage training process. | The gate should be introduced through fine-tuning or staged adaptation, not by redesigning the full training recipe from scratch. |
| MapTracker reports both standard mAP and consistency-aware metrics. | The MVP should protect clean performance and use consistency-oriented contradiction metrics rather than claiming improvement from mAP alone. |

The public repository also confirms the relevant top-level structure: the code lives primarily under `plugin/`, with `docs/` and `tools/` used for environment setup and runs. For this project, the main modifications should stay concentrated in the transformer utilities, vector memory utilities, and the training / loss wrapper.

## 5. Architecture and exact modification point

The most important architectural requirement is that suppression happens before historical memory has already been fused. If you only have access to an already aggregated historical output, you are rebuilding the wrong method.

For each VEC decoder layer, use the following logical processing order:

1. PropMLP propagates historical queries from frame t-1 into frame t and new candidate queries are appended.
2. Vector self-attention updates the current query set.
3. BEV-to-vector cross-attention produces `q_cur`, the current-evidence-updated query tensor.
4. The MVP temporal memory gate reads `q_cur` and the selected vector memory bank, computes `alpha(q, t_slot)`, and scales the historical values before historical attention.
5. Historical vector-memory attention consumes the gated values and untouched keys.
6. Residual output is handed to the next decoder layer.

**Authoritative insertion point:** inside each `MapTransformerLayer`, after BEV-to-vector cross-attention and before historical vector-memory attention.

## 6. Authoritative tensor contract

The MVP must standardize the logical interface around the gate even if the in-code tensor ordering differs slightly. Future debugging becomes much easier if these contracts are written down and enforced.

| Tensor / mask | Logical shape | Meaning |
| --- | --- | --- |
| `q_cur` | `[Q, B, D]` | Current query tensor after BEV-to-vector cross-attention in the current decoder layer. |
| `K_mem` | `[Q, B, H, T, d]` or equivalent | Selected historical keys before historical attention. |
| `V_mem` | `[Q, B, H, T, d]` or equivalent | Selected historical values before historical attention. |
| `valid_mem` | `[Q, B, T]` | Whether the slot exists and is valid for that query. |
| `propagated_mask` | `[Q, B]` | True only for queries propagated from prior frame state. |
| `new_mask` | `[Q, B]` | True for newly initialized queries created in the current frame. |
| `pad_mask` | `[Q, B]` | True for padding positions introduced by batching. |
| `delta_t_int` | `[Q, B, T]` | Raw frame offset for each selected historical slot. |
| `age_rank_norm` | `[Q, B, T]` | Relative age rank of a slot among valid history slots for that query. |
| `slot_corrupt_mask` | `[Q, B, T]` | Synthetic label marking whether the read-view slot is stale-corrupted in the current clip. |
| `alpha_soft` | `[Q, B, T]` | Scalar gate value per eligible query-slot pair. |
| `V_mem_soft` | same as `V_mem` | Value tensor after alpha-based suppression and before historical attention. |

## 7. Query-state semantics and masking rules

Mask semantics are non-negotiable. They are not decorative bookkeeping. They determine which queries may consume history and which losses are valid.

- A propagated query may use temporal history and may receive gate supervision if valid memory exists.
- A new query must not consume historical vector memory in the MVP. Historical output for such queries is forced to zero.
- A padded query is ignored everywhere: no history consumption, no gate supervision, no logging contribution.
- No fallback such as `query_memory.sum() == 0` is allowed. Eligibility must be determined explicitly by masks and valid memory metadata.

Define the masks exactly as follows:

```python
eligible_query_mask = propagated_mask & (~pad_mask)
eligible_slot_mask = eligible_query_mask.unsqueeze(-1) & valid_mem
```

For any location where `eligible_slot_mask` is false, `alpha_soft` must be zeroed and the gated historical contribution must evaluate to zero.

## 8. Final gate design

### 8.1 Gate granularity

Use one scalar gate per query-slot pair: `alpha[q, b, t]`. Broadcast that scalar across all attention heads when scaling the value tensor. This preserves temporal selectivity while avoiding the engineering and attribution burden of independent per-head gating.

### 8.2 Feature inputs

The gate input should be deliberately small, stable, and interpretable. The MVP uses disagreement and age, not an auxiliary reliability branch.

- `u_cur`: `LayerNorm(q_cur)`, expanded to the slot dimension.
- `u_key`: `LayerNorm` of a slotwise historical key summary.
- `u_val`: `LayerNorm` of a slotwise historical value summary.
- `cos_key`: cosine similarity between `u_cur` and `u_key`.
- `l2_val`: normalized L2 distance between `u_cur` and `u_val`.
- `pe_age`: sinusoidal encoding of `delta_t_int` with a small fixed dimension such as 8.
- `age_rank_norm`: a scalar recency indicator that helps separate recent preserved slots from older stale ones under C-tail.

If `K_mem` and `V_mem` are stored per head, first reconstruct a slotwise summary before feeding them to the gate. The simplest method is to reshape the full multi-head tensor back to `[Q, B, T, D]` and apply `LayerNorm` there. Avoid inventing a second complicated summary network in the MVP.

### 8.3 Gate equation

Recommended feature construction:

```python
u_cur_t    = LN(q_cur).unsqueeze(2).expand(Q, B, T, D)
u_key      = LN(K_slot)
u_val      = LN(V_slot)
cos_key    = cosine(u_cur_t, u_key)                # [Q, B, T, 1]
l2_val     = ||u_cur_t - u_val||_2 / sqrt(D)       # [Q, B, T, 1]
pe_age     = sinusoidal(delta_t_int, d_pe=8)       # [Q, B, T, 8]
gate_in    = concat(u_cur_t, u_key, u_val, cos_key, l2_val, pe_age, age_rank_norm)
alpha_soft = sigmoid(MLP(gate_in))                 # [Q, B, T]
```

Recommended gate MLP: `Linear(input_dim, 64) -> GELU -> Linear(64, 1) -> Sigmoid`. This is intentionally small. Do not add bilinear disagreement matrices, confidence heads, or extra projections unless the MVP is already stable.

### 8.4 Action mechanism

Use value scaling only:

```python
alpha_broadcast = alpha_soft.unsqueeze(2).unsqueeze(-1)  # [Q, B, 1, T, 1]
V_mem_soft = alpha_broadcast * V_mem
```

Keys remain unchanged. There is no additive logit bias and no hard thresholding in training. The goal is to establish a smooth, stable suppression mechanism first.

## 9. Corruption protocol

The corruption protocol is part of the method, not a training detail. The gate will not learn to reject stale memory unless it sees controlled negative examples.

### 9.1 Allowed corruption modes in the MVP

| Mode | Definition | Why it is included |
| --- | --- | --- |
| Clean | No stale corruption. Standard selected temporal bank is read. | Prevents trivial always-closed behavior and preserves clean-data behavior. |
| C-full | All valid historical slots in the selected read bank are replaced by staler-than-intended slots. | Teaches full-bank stale rejection. |
| C-tail | Only older valid slots are replaced by staler-than-intended slots while the most recent valid slots remain clean. | Teaches the selective suppression that slotwise gating is meant to provide. |

### 9.2 Corruption semantics

- Corruption is applied on the read path only. The canonical memory bank remains clean.
- Corrupted keys / values shown to the current frame are local perturbations for that forward pass only.
- The corrupted read tensors must not be written back into the canonical memory state.
- Each training clip has exactly one corruption state: clean, C-full, or C-tail.
- Corruption starts only after an interior onset `t_inj` so that earlier frames in the clip remain clean.

Recommended clip probabilities for the MVP: 60% clean, 20% C-full, 20% C-tail.

### 9.3 Stale offset policy

Use a small, supported stale-offset set at first. Recommended training stale offsets: `{4, 8}`. Only expand later if the memory buffer and bookkeeping are already stable.

For any query-slot where the requested stale source does not exist, do not fabricate a target. Mark that slot as ineligible for corruption loss for that forward pass.

### 9.4 Slot labels

The corruption engine must return `slot_corrupt_mask[q, b, t]`. This is the authoritative label for gate supervision in the MVP.

- Under C-full, every valid history slot after onset is labeled corrupted.
- Under C-tail, only the stale older subset is labeled corrupted; preserved recent slots are labeled clean.
- Under clean clips, slots are not treated as negative examples for close supervision; only optional weak open regularization may be used.

## 10. Loss design

The MVP keeps the MapTracker base objective and adds a small, direct gate supervision loss. It does not introduce geometry-localized routing, masked transformation loss, or ranking losses.

### 10.1 Base task loss

```python
L_base = L_BEV + L_track + lambda_trans * L_trans
```

Keep the host loss unchanged in the MVP. Because the MVP uses stale-bank substitution rather than explicit transformation corruption, you do not need to mask `L_trans` in the first build.

### 10.2 Gate supervision

Use direct supervision only where labels are truly known from the corruption engine.

```python
affected_slot_mask  = eligible_slot_mask & slot_corrupt_mask
preserved_slot_mask = eligible_slot_mask & (~slot_corrupt_mask)
```

Recommended gate loss design:

```python
L_close = BCE(alpha_soft[affected_slot_mask], 0)
L_open  = BCE(alpha_soft[preserved_recent_mask], 1)
L_clean = BCE(alpha_soft[most_recent_valid_slot_on_clean], 1)  # optional, low weight
L_gate  = lambda_close * L_close + lambda_open * L_open + lambda_clean * L_clean
```

Recommended starting weights: `lambda_close = 1.0`, `lambda_open = 0.5`, `lambda_clean = 0.1`.

Why not supervise all clean slots with target 1? Because alpha is a control variable, not a semantic label. Over-supervising all clean slots pushes the model toward always-open behavior and reduces flexibility. Supervise the slots whose meaning is clear from the synthetic protocol.

## 11. Code modification map

This section maps the MVP to concrete code areas. File names reflect the earlier code guide and the public repo structure, but you must verify exact line numbers against your local checkout.

### 11.1 transformer_utils / MapTransformer.py

- Add a new `SlotwiseTemporalGate` module. Keep it simple: one MLP, no bilinear matrices, no confidence branch, no hard skip.
- Instantiate one gate module per `MapTransformerLayer`.
- Modify the forward path so the gate is called after BEV-to-vector cross-attention and before historical vector-memory attention.
- Pass `q_cur`, `K_mem`, `V_mem`, `valid_mem`, `delta_t_int`, `age_rank_norm`, and the query-state masks into the gate.
- Return `alpha_soft` for loss computation and logging, and `V_mem_soft` for historical attention.
- Do not modify the BEV multi-point attention operator in the MVP.

### 11.2 mapers / vector_memory.py

- Keep the canonical memory write path unchanged.
- Build the normal selected temporal bank exactly as MapTracker would for clean inference.
- Add a corrupted read-view constructor for C-full and C-tail.
- Return `valid_mem`, `delta_t_int`, `age_rank_norm`, and `slot_corrupt_mask` together with the selected keys and values.
- Ensure corruption happens only for the local read bank, never for the stored canonical buffer.
- If a stale replacement source is missing, mark the slot unsupervised for gate loss rather than silently substituting an invalid tensor.

### 11.3 criterion / training wrapper

- Compute `L_close`, `L_open`, and optional `L_clean` from `alpha_soft` and the slot masks.
- Log alpha statistics needed for collapse diagnosis.
- Keep the MapTracker base losses intact.
- Do not add `geom_fail_mask`, `prev_query_L_trans` routing, masked `L_trans`, or rank losses in the MVP.

### 11.4 config files and experiment wiring

- Add corruption mode probabilities and stale offsets.
- Add gate loss weights.
- Add stage-wise freeze settings for gate warmup and decoder co-adaptation.
- Add evaluation flags for clean validation and contradiction-suite validation.
- Add a switch to run a corruption-trained no-gate baseline with the exact same data / schedule settings.

## 12. Reference pseudocode

The pseudocode below is the recommended logical starting point. It is intentionally simple and should be adapted to your exact tensor ordering and module interfaces.

```python
class SlotwiseTemporalGate(nn.Module):
    def __init__(self, embed_dims=256, d_pe=8):
        super().__init__()
        self.embed_dims = embed_dims
        self.d_pe = d_pe
        gate_in_dim = embed_dims * 3 + 2 + d_pe + 1  # u_cur, u_key, u_val, cos, l2, PE, age_rank
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, q_cur, K_slot, V_slot, delta_t_int, age_rank_norm, eligible_slot_mask):
        # q_cur: [Q, B, D]
        # K_slot: [Q, B, T, D] reconstructed slotwise summary
        # V_slot: [Q, B, T, D] reconstructed slotwise summary
        u_cur = F.layer_norm(q_cur, (q_cur.size(-1),))
        u_key = F.layer_norm(K_slot, (K_slot.size(-1),))
        u_val = F.layer_norm(V_slot, (V_slot.size(-1),))

        u_cur_t = u_cur.unsqueeze(2).expand_as(u_key)
        cos_key = F.cosine_similarity(u_cur_t, u_key, dim=-1).unsqueeze(-1)
        l2_val = (u_cur_t - u_val).norm(dim=-1).div(math.sqrt(u_cur.size(-1))).clamp(max=1.0).unsqueeze(-1)
        pe_age = sinusoidal_encode(delta_t_int, d_pe=self.d_pe)
        age_rank = age_rank_norm.unsqueeze(-1)

        gate_in = torch.cat([u_cur_t, u_key, u_val, cos_key, l2_val, pe_age, age_rank], dim=-1)
        alpha_soft = self.gate_mlp(gate_in).squeeze(-1)
        alpha_soft = torch.where(eligible_slot_mask, alpha_soft, torch.zeros_like(alpha_soft))
        return alpha_soft

# Historical value suppression path
alpha_broadcast = alpha_soft.unsqueeze(2).unsqueeze(-1)  # [Q, B, 1, T, 1]
V_mem_soft = alpha_broadcast * V_mem  # V_mem: [Q, B, H, T, d]
# Historical attention then consumes original keys K_mem and gated values V_mem_soft
```

## 13. Training schedule

Do not restart the full host model from scratch if you already have a stable MapTracker checkpoint. The gate should enter through staged adaptation.

| Phase | Trainable modules | Frozen modules | Goal |
| --- | --- | --- | --- |
| Phase A - baseline checkpoint | None; load stable MapTracker checkpoint | N/A | Establish the exact host baseline and evaluation pipeline. |
| Phase B - gate warmup | Gate modules only | Backbone, BEV encoder, PropMLP, existing decoder attention | Teach the gate to separate corrupted slots from preserved slots without destabilizing the host model. |
| Phase C - decoder co-adaptation | Gate modules + VEC decoder layers + historical attention projections | Backbone and BEV encoder | Recover downstream task performance while preserving contradiction gains. |
| Phase D - optional short joint tune | All modules only if necessary | None | Use only if the system is stable and time allows; otherwise skip. |

Recommended practical schedule: 5 to 10 epochs for gate warmup, then 10 to 15 epochs for decoder co-adaptation. Treat these as starting points, not immutable numbers.

## 14. Baselines and experiment matrix

The baseline set must be strong enough to isolate the effect of gating but small enough to finish.

| ID | Model | Required | Purpose |
| --- | --- | --- | --- |
| B0 | Standard MapTracker | Yes | Reference clean baseline and host performance anchor. |
| B1 | Corruption-trained no-gate MapTracker | Yes | Separates the effect of corruption-trained data from the effect of the gate itself. |
| B2 | MVP slotwise gate | Yes | Primary method. |
| B3 | Bankwise gate | Optional | Useful only if time permits and you want a direct slotwise-versus-bankwise comparison. |

**Parity rule:** B1 and B2 must use the same corruption sampler, the same number of optimizer steps, the same checkpoint-selection rule, and the same evaluation protocol. Otherwise the causal comparison is weak.

## 15. Evaluation plan

### 15.1 Clean validation

- Standard mAP on the validation set.
- C-mAP if your pipeline already exposes it.
- Optional runtime / FPS check only at the end.

Interpretation rule: clean performance may dip slightly during corruption-aware training, but the gated model should not suffer a large or unstable collapse relative to B0 and B1.

### 15.2 Contradiction-suite validation

- Evaluate on C-full and C-tail.
- Measure stale persistence time: how many frames false stale geometry survives after contradiction onset.
- Measure cumulative stale false-positive polyline length over time.
- Measure alpha separation under C-tail: corrupted older slots should receive lower alpha than preserved recent slots.

The contradiction suite is the main evaluation arena for the method. Standard mAP alone cannot validate the project claim.

### 15.3 Qualitative outputs

- One clean scene comparison: B0 vs B1 vs B2.
- One C-full scene showing stale-history cleanup.
- One C-tail scene showing selective preservation of recent slots and rejection of stale older slots.

## 16. Logging and must-have tests

Do not recreate the full v3 diagnostics bureaucracy. Keep the tests and logs that prevent silent invalidation.

### 16.1 Required logs

- Mean alpha on affected corrupted slots.
- Mean alpha on preserved recent slots.
- Mean alpha on clean clips or on the most recent valid clean slot.
- Fraction of batches with non-empty `affected_slot_mask`.
- Historical path strength ratio on clean validation: `||mem_out|| / ||q_cur||` to detect over-suppression.

### 16.2 Required tests

- Dimension smoke test: one forward pass succeeds with all added tensors and masks.
- No-history test: new and padded queries produce zero historical contribution.
- One-alpha parity test: alpha = 1 reproduces the ungated historical path.
- Zero-alpha suppression test: alpha = 0 removes historical contribution.
- Corrupted-read isolation test: local read corruption does not modify canonical memory.
- C-tail selectivity test: after warmup, corrupted older slots get lower alpha than preserved recent slots on a controlled synthetic example.

## 17. Failure modes and responses

| Failure mode | Likely symptom | Response |
| --- | --- | --- |
| Always-open gate | Affected slots and preserved slots have very similar high alpha. | Increase `lambda_close` modestly, verify `slot_corrupt_mask` routing, and confirm corrupted samples actually reach the model. |
| Always-closed gate | Clean or preserved recent slots have low alpha and clean metrics collapse. | Reduce `lambda_close`, increase `lambda_open` slightly, and verify clean / preserved recent supervision exists. |
| No useful alpha separation | Alpha distributions overlap heavily after warmup. | Check corruption construction, `eligible_slot_mask`, and whether the gate is accidentally reading already fused history instead of pre-fusion slots. |
| Historical path effectively dead | `mem_out` is tiny even on clean data. | Lower close pressure, verify alpha masking logic, and compare one-alpha parity to the baseline path. |
| Corruption contamination bug | Performance drifts unpredictably across later frames and stored memory appears corrupted. | Audit the read / write split and verify the canonical memory bank is never overwritten by the corrupted read view. |

## 18. Write-up discipline and claim boundaries

The MVP should make a disciplined claim, not a grand claim. If the method works, the allowed conclusion is narrow and still meaningful: a slotwise pre-fusion temporal gate can suppress stale historical vector memory under the defined synthetic contradiction protocol without materially harming clean mapping quality.

- Do not claim perception confidence if the reliability branch is absent.
- Do not claim the method repairs wrong memory; it only suppresses temporal contribution.
- Do not claim general real-world scene-change robustness from synthetic stale corruption alone.
- Do not describe the gate as a contradiction oracle; it is a learned suppression mechanism trained under a controlled protocol.

## 19. Implementation order of operations

Follow this order. Do not jump ahead and wire every deferred feature at once.

7. Verify the current MapTracker baseline training / evaluation path and checkpoint loading.
8. Locate the exact pre-fusion point in `MapTransformerLayer` where historical vector-memory attention consumes `K_mem` and `V_mem`.
9. Add explicit propagated / new / pad mask outputs if they are not already exposed cleanly.
10. Extend `vector_memory.py` to return `valid_mem`, `delta_t_int`, `age_rank_norm`, and a clean selected history bank.
11. Implement the local corrupted read-view path for C-full and C-tail and return `slot_corrupt_mask`.
12. Implement `SlotwiseTemporalGate` and wire it into the pre-fusion path.
13. Add gate loss computation and alpha logging.
14. Run dimension and parity tests before any long training run.
15. Train B1 and B2 under parity, compare contradiction metrics, and only then consider deferred additions.

## 20. Deferred roadmap after the MVP

Only pursue these if the MVP is stable and gives a real signal:

- Run a small bankwise ablation if you still need a stronger slotwise claim.
- Add a small natural inconsistency subset to test bounded transfer.
- Test additive logit bias only if value scaling already works.
- Only after that, consider the BEV reliability branch.
- Leave translation / rotation corruption and geometry-localized routing for last.

## Appendix A. Minimal config fields to add

```yaml
model:
  use_temporal_gate: true
  temporal_gate:
    embed_dims: 256
    d_pe: 8
    hidden_dim: 64
    share_across_heads: true

train:
  corruption:
    p_clean: 0.60
    p_c_full: 0.20
    p_c_tail: 0.20
    stale_offsets: [4, 8]
    use_read_path_corruption_only: true
  gate_loss:
    lambda_close: 1.0
    lambda_open: 0.5
    lambda_clean: 0.1

finetune:
  phase_b_gate_only: true
  phase_c_unfreeze_vec_decoder: true
  freeze_backbone: true
  freeze_bev_encoder: true

eval:
  run_clean_metrics: true
  run_contradiction_suite: true
  contradiction_modes: [c_full, c_tail]
  save_alpha_stats: true
```

## Appendix B. Final one-paragraph project description

This project modifies MapTracker by inserting a pre-fusion slotwise temporal memory gate into the VEC decoder. For each propagated query and each selected historical memory slot, the gate compares current query evidence against historical key / value summaries plus temporal age features and outputs a scalar alpha that suppresses the historical value contribution before historical attention. The system is trained with read-path-only synthetic stale-memory corruption using clean clips, full-bank stale corruption, and tail-only stale corruption, and evaluated against a standard MapTracker baseline and a corruption-trained no-gate baseline. The intended claim is narrow: under a controlled stale-memory contradiction protocol, pre-fusion slotwise temporal gating can improve contradiction recovery without materially damaging clean vectorized HD mapping quality.

## MVP evaluation commands (clean + contradiction suite)

Clean eval (default corruption control is clean):

```bash
python tools/test.py <CONFIG> <CHECKPOINT> --eval --work-dir <WORK_DIR>
```

Condition-specific contradiction eval (example: C-tail, stale_offset=8):

```bash
python tools/test.py <CONFIG> <CHECKPOINT> --eval --work-dir <WORK_DIR>/c_tail_offset8 \
  --memory-corruption-mode c_tail \
  --memory-stale-offset 8 \
  --memory-c-tail-keep-recent 1 \
  --memory-corruption-onset <ONSET> \
  --condition-tag c_tail_offset8_onset<ONSET>
```

Full contradiction suite (mode in `{c_full,c_tail}`, stale_offset list `[4,8]`):

```bash
python tools/tracking/run_contradiction_suite.py <CONFIG> <CHECKPOINT> \
  --work-root <WORK_DIR>/contradiction_suite \
  --modes c_full c_tail \
  --stale-offsets 4 8 \
  --onset <ONSET>
```

Output locations:

- Per-condition eval artifacts are written under `<WORK_DIR>/contradiction_suite/<mode>_offset<k>_onset<t>/`.
- Standard prediction artifact: `submission_vector.json`.
- Condition metadata: `condition_meta.json`.
- Contradiction metrics: `contradiction_metrics.json`.
- Suite aggregate summary + qualitative scene pointers: `contradiction_suite_summary.json`.

## References and source basis

- Original Project Proposal by Yousef Kassem.
- Disagreement-Gated Decoder Overview.
- Code Implementation for Disagreement Gated Decoder.
- TechSpec_v3.
- Chen et al., *MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping*, ECCV 2024, and the woodfrog/maptracker public repository README on GitHub.
