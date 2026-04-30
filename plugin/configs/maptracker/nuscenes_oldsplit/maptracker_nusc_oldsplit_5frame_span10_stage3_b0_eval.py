# B0 evaluation config: vanilla MapTracker (original published checkpoint, no MVP gate).
#
# Inherits all architecture / data / schedule settings from Stage-3 Joint Finetune and
# applies two eval-relevant overrides:
#
#   1. model.head_cfg.transformer.decoder.transformerlayers.temporal_gate_cfg.enabled = False
#      The gate module is still instantiated (SlotwiseTemporalGate is always constructed),
#      but with enabled=False its forward() early-returns unmodified embeddings (all-ones
#      alpha).  This is required because the official B0 checkpoint has no
#      temporal_gate.gate_mlp.* weights; without this override those weights stay at
#      random initialisation and corrupt every evaluation forward pass.
#
#   2. model.mvp_temporal_gate_cfg supervision / corruption flags are set to no-op values.
#      These fields are only consumed during training, but resetting them here makes the
#      config semantically consistent: B0 was never trained with the MVP gate.
#
# eval_corruption_cfg is NOT set here.  tools/test.py:apply_eval_corruption_overrides()
# injects it at eval time from --memory-corruption-mode / --memory-stale-offset / etc.
# when running corruption evaluations.  It handles an absent mvp_temporal_gate_cfg
# gracefully (returns {} and creates the sub-dict as needed).

_base_ = ['maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py']

model = dict(
    # -------------------------------------------------------------------------
    # Override 1: neutralise gate training flags (no-ops during eval, but keep
    # the config semantically consistent with a gate-free pretrained model).
    # -------------------------------------------------------------------------
    mvp_temporal_gate_cfg=dict(
        gate_supervision_enabled=False,
        corruption_probs=dict(clean=1.0, c_full=0.0, c_tail=0.0),
    ),
    # -------------------------------------------------------------------------
    # Override 2: disable gate execution in the decoder transformer layers.
    # Only temporal_gate_cfg is changed; all other transformerlayers keys
    # (type, attn_cfgs, ffn_cfgs, etc.) are preserved from the base config
    # via mmcv's recursive dict merge.
    # -------------------------------------------------------------------------
    head_cfg=dict(
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    temporal_gate_cfg=dict(
                        enabled=False,
                        hidden_dims=64,
                    ),
                ),
            ),
        ),
    ),
)
