# Divergence Register (Initial)

This register tracks mismatches between `docs/mvp_spec.md` intent and observed implementation behavior.

| Divergence ID | Related requirement(s) | Severity | Type | Evidence | Description | Proposed resolution |
|---|---|---|---|---|---|---|
| DIV-001 | RQ-ARCH-001, RQ-TENSOR-001 | Medium | D3 (implementation constraint) | `plugin/models/transformer_utils/MapTransformer.py` lines around gate assertion | Gate path asserts `q_len == 1`, which narrows the generic tensor contract implied by MVP (`[Q,B,D]` generality). | Update `final_implemented_spec.md` to explicitly document packed single-query memory-branch semantics; optionally generalize gate for `Q>1` in future work. |
| DIV-002 | RQ-GATE-003 | Medium | D4 (semantic drift) | `plugin/models/transformer_utils/MapTransformer.py` gate feature construction | MVP text recommends separate `u_key`, `u_val`, sinusoidal age encoding, and age-rank; current implementation uses `q` + `mem` with scalar features (`cos`, `l2`, `valid_mem`, normalized `delta_t`, `age_rank`) and no dedicated sinusoidal age block. | Document as implemented feature set in `final_implemented_spec.md`; consider optional PE-age ablation if needed. |
| DIV-003 | RQ-MASK-001, RQ-MASK-002 | High | D2 (partial semantics) | `plugin/models/transformer_utils/MapTransformer.py` eligible mask assembly | Eligibility currently uses `valid_query_mask` + memory validity for tracked indices; explicit propagated/new/pad mask taxonomy from MVP is not surfaced as first-class tensors in this module. | Capture true effective semantics in final spec and add explicit propagated/new/pad outputs if strict MVP parity is required. |
| DIV-004 | RQ-TEST-* validation confidence | Medium | D6 (verification gap) | `python -m pytest -q tests/test_temporal_gate_mvp.py` output (`11 skipped`) | Unit tests exist, but this environment skipped all of them; behavioral runtime validation remains incomplete here. | Mark current evidence as static-presence only; re-run in dependency-complete environment and record outcomes in this register. |
| DIV-005 | RQ-TENSOR-002 | Medium | D4 (contract drift) | `plugin/models/transformer_utils/MapTransformer.py` memory branch (`mem_embeds`, `mem_values`) | MVP contract discusses separate `K_mem` and `V_mem` tensors (potentially with head axes), but implementation operates on slotwise memory embeddings and gated values without explicit multi-head key/value tensor contract at gate interface. | Document the actual contract in `final_implemented_spec.md` and avoid claiming explicit per-head K/V gating unless implemented. |

## Notes
- This file is intentionally conservative: divergences are not failures by default; they indicate where the as-built system differs from normative MVP wording.
- Additional divergences should be appended during Phase 2 and linked back to requirement IDs.
