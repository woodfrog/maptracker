# Cooldown config: 5-epoch LR-tail continuation from iter_89148 checkpoint.
#
# Context
# -------
# B1/B2 Stage-3 training stopped at iter 89,148 (53 % of the 167,808-iter
# cosine schedule).  At that point the main-param LR was ~2.270e-4 (backbone
# ~2.27e-5) — the cosine midpoint — so the entire low-LR refinement tail was
# missed.  This config resumes *weights-only* from that checkpoint and applies
# a fresh 5-epoch cosine schedule that starts at the exact midpoint LR and
# decays to the same min_lr as the original run.
#
# What is intentionally left OUT of this file (passed at submission time)
# -----------------------------------------------------------------------
#   load_from          – provided as --base-checkpoint to run_b1_b2.sh, which
#                        appends load_from=<path> automatically when --resume
#                        is absent.
#   num_gpus           – hardware-dependent; passed as cfg-option so the same
#   data.samples_per_gpu  config works for both the 8-GPU (preferred) and
#   fp16.loss_scale       4-GPU (fallback) submission variants.
#   optimizer.lr       – passed via cfg-options so the exact midpoint value is
#   lr_config.*           visible in the submission scripts rather than buried
#                        in a config file.  Avoids silent mismatches if the
#                        config is reused for a different resume point.

_base_ = [
    './maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py',
]

# 5 epochs × 1748 iters/epoch (27968 samples / (8 GPUs × 2 batch_size) = 1748)
# The iters-per-epoch value itself is fixed regardless of GPU count because
# num_gpus/samples_per_gpu only affect the num_iters_per_epoch *variable* in
# the base config, not runner.max_iters which we override directly here.
runner = dict(type='MyRunnerWrapper', max_iters=8740)

# Save a checkpoint after every epoch (1748 iters) so we can compare per-epoch
# metrics and recover from any job interruption without losing progress.
checkpoint_config = dict(interval=1748)

# Run a single full validation pass only at the end of the cooldown.
# Skipping mid-run evals saves ~1.5 h of wall time across the 5 epochs.
evaluation = dict(interval=8740)
