"""Lightweight static validator for Milestone 4 wiring without torch runtime."""
from pathlib import Path

checks = {
    'maptracker_losses': ('plugin/models/mapers/MapTracker.py', ['L_close', 'L_open', 'alpha_mean_affected', 'affected_batch_fraction']),
    'transformer_alpha': ('plugin/models/transformer_utils/MapTransformer.py', ['batch_alpha_soft_dict', 'batch_v_mem_soft_dict']),
}

ok = True
for name, (fp, needles) in checks.items():
    text = Path(fp).read_text()
    miss = [n for n in needles if n not in text]
    if miss:
        ok = False
        print(f'[FAIL] {name}: missing {miss}')
    else:
        print(f'[PASS] {name}')

if not ok:
    raise SystemExit(1)
print('Milestone 4 static checks passed.')
