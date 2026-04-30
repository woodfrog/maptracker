"""
vis_alpha_heatmap.py — Visualize per-slot gate alpha values from alpha_per_frame.pkl.

Produces two figure types:
  1. Per-frame heatmap: T_mem × N_valid_tracks grid coloured by alpha.
     Corrupted slots are outlined in red; preserved-recent in green.
  2. Temporal trace: per-scene line plot of α_affected and α_preserved_recent
     across all frames.

Usage:
  python tools/visualization/vis_alpha_heatmap.py \\
      --b1-pkl  CurrentB1B2Results/b1_contra_89148/c_full_offset1_onset0/alpha_per_frame.pkl \\
      --b2-pkl  CurrentB1B2Results/b2_contra_89148/c_full_offset1_onset0/alpha_per_frame.pkl \\
      --out-dir qualitative_outputs/alpha_heatmaps/cfull_offset1 \\
      [--scenes scene-0003 scene-0012] \\
      [--max-tracks 32] \\
      [--no-per-frame]

Output layout:
  <out-dir>/
    <scene>/
      b1_frame<N>.png   b2_frame<N>.png  (per-frame heatmaps, if --no-per-frame not set)
      b1_vs_b2_frame<N>.png              (side-by-side per-frame comparison panels)
      temporal_trace.png                 (α_affected / α_preserved_recent vs frame index)
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize per-slot gate alpha heatmaps')
    parser.add_argument('--b1-pkl', required=True,
                        help='Path to alpha_per_frame.pkl for B1')
    parser.add_argument('--b2-pkl', required=True,
                        help='Path to alpha_per_frame.pkl for B2')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory')
    parser.add_argument('--scenes', nargs='+', default=None,
                        help='Limit to these scene names')
    parser.add_argument('--max-tracks', type=int, default=48,
                        help='Maximum number of tracks to show per frame (truncated by instance index)')
    parser.add_argument('--no-per-frame', action='store_true',
                        help='Skip per-frame PNG output (only produce temporal trace)')
    parser.add_argument('--dpi', type=int, default=100,
                        help='DPI for saved figures')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def group_by_scene(records):
    """Group list[dict] by scene_name, sorted by local_idx within each scene."""
    scenes = defaultdict(list)
    for rec in records:
        if rec is None:
            continue
        scenes[rec['scene_name']].append(rec)
    for name in scenes:
        scenes[name].sort(key=lambda r: r['local_idx'])
    return dict(scenes)


# ---------------------------------------------------------------------------
# Per-frame heatmap
# ---------------------------------------------------------------------------

def _draw_alpha_heatmap(ax, alpha, corrupt, eligible, valid_mem, title, max_tracks):
    """Draw a T_mem × N_tracks heatmap on `ax`.

    alpha:    [N_tracks, T_mem] float
    corrupt:  [N_tracks, T_mem] bool — injected stale slots
    eligible: [N_tracks, T_mem] bool — slots eligible for corruption supervision
    valid_mem:[N_tracks, T_mem] bool — non-padding slots
    """
    n_tracks, t_mem = alpha.shape
    if n_tracks > max_tracks:
        alpha = alpha[:max_tracks]
        corrupt = corrupt[:max_tracks]
        eligible = eligible[:max_tracks]
        valid_mem = valid_mem[:max_tracks]
        n_tracks = max_tracks

    # Mask padding (invalid) slots as NaN so they show grey
    display = alpha.copy().astype(float)
    display[~valid_mem] = np.nan

    # Transpose: rows = mem slots (T_mem), cols = tracks (N)
    img = display.T  # [T_mem, N_tracks]

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color='#cccccc')  # grey for padding
    im = ax.imshow(img, aspect='auto', vmin=0.0, vmax=1.0,
                   cmap=cmap, interpolation='nearest', origin='lower')

    # Outline corrupted slots (red border per cell)
    corrupt_T = corrupt.T  # [T_mem, N_tracks]
    eligible_T = eligible.T

    for ti in range(t_mem):
        for ni in range(n_tracks):
            if not valid_mem[ni, ti]:
                continue
            if corrupt_T[ti, ni] and eligible_T[ti, ni]:
                rect = mpatches.FancyBboxPatch(
                    (ni - 0.5, ti - 0.5), 1.0, 1.0,
                    boxstyle='square,pad=0', linewidth=1.5,
                    edgecolor='red', facecolor='none')
                ax.add_patch(rect)

    ax.set_xlabel('Track index', fontsize=8)
    ax.set_ylabel('Memory slot (age)', fontsize=8)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='α')


def render_frame_comparison(rec_b1, rec_b2, out_path, max_tracks, dpi):
    """Side-by-side B1 | B2 heatmap for a single frame."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    scene = rec_b1['scene_name']
    local_idx = rec_b1['local_idx']
    mode = rec_b1.get('mode', '?')

    for ax, rec, label in zip(axes, [rec_b1, rec_b2], ['B1 (gate-open)', 'B2 (trained gate)']):
        _draw_alpha_heatmap(
            ax,
            rec['alpha'], rec['corrupt'], rec['eligible'], rec['valid_mem'],
            title=f'{label} | {scene} frame {local_idx:03d} | mode={mode}',
            max_tracks=max_tracks,
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color='red', label='Corrupted+eligible slot'),
        mpatches.Patch(color='#cccccc', label='Padding (no mem)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def render_single_heatmap(rec, out_path, label, max_tracks, dpi):
    fig, ax = plt.subplots(figsize=(6, 4))
    mode = rec.get('mode', '?')
    title = f'{label} | {rec["scene_name"]} frame {rec["local_idx"]:03d} | mode={mode}'
    _draw_alpha_heatmap(ax, rec['alpha'], rec['corrupt'], rec['eligible'], rec['valid_mem'],
                        title=title, max_tracks=max_tracks)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Temporal trace
# ---------------------------------------------------------------------------

def _compute_trace(records):
    """Return (frame_indices, alpha_affected_list, alpha_preserved_list)."""
    frames, affected, preserved = [], [], []
    for rec in records:
        if rec is None:
            continue
        alpha = rec['alpha']      # [N_tracks, T_mem]
        corrupt = rec['corrupt']  # [N_tracks, T_mem]
        eligible = rec['eligible']
        valid_mem = rec['valid_mem']
        age_rank = rec['age_rank']  # [N_tracks, T_mem]

        affected_mask = eligible & corrupt
        alpha_aff = float(alpha[affected_mask].mean()) if affected_mask.any() else float('nan')

        recency_ref = age_rank.max(axis=1, keepdims=True)
        preserved_mask = valid_mem & (~corrupt) & (age_rank >= recency_ref)
        alpha_pres = float(alpha[preserved_mask].mean()) if preserved_mask.any() else float('nan')

        frames.append(rec['local_idx'])
        affected.append(alpha_aff)
        preserved.append(alpha_pres)
    return np.array(frames), np.array(affected), np.array(preserved)


def render_temporal_trace(b1_records, b2_records, out_path, scene_name, dpi):
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    for ax, records, label, color_aff, color_pres in [
        (axes[0], b1_records, 'B1 (gate-open)', '#d62728', '#2ca02c'),
        (axes[1], b2_records, 'B2 (trained gate)', '#d62728', '#2ca02c'),
    ]:
        frames, affected, preserved = _compute_trace(records)
        ax.plot(frames, affected, color=color_aff, lw=1.5, label='α_affected (stale/corrupted slots)')
        ax.plot(frames, preserved, color=color_pres, lw=1.5, linestyle='--',
                label='α_preserved_recent (fresh slots)')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('Mean α', fontsize=9)
        ax.set_title(f'{label} — {scene_name}', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color='grey', lw=0.8, linestyle=':')

    axes[1].set_xlabel('Frame index (local_idx)', fontsize=9)
    fig.suptitle(f'Gate α temporal trace — {scene_name}', fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f'[vis_alpha_heatmap] Loading B1 pkl: {args.b1_pkl}')
    b1_all = load_pkl(args.b1_pkl)
    print(f'[vis_alpha_heatmap] Loading B2 pkl: {args.b2_pkl}')
    b2_all = load_pkl(args.b2_pkl)

    b1_scenes = group_by_scene(b1_all)
    b2_scenes = group_by_scene(b2_all)

    target_scenes = args.scenes if args.scenes is not None else sorted(set(b1_scenes) & set(b2_scenes))
    print(f'[vis_alpha_heatmap] Processing {len(target_scenes)} scenes: {target_scenes}')

    for scene in target_scenes:
        if scene not in b1_scenes:
            print(f'  [skip] {scene} not in B1 pkl')
            continue
        if scene not in b2_scenes:
            print(f'  [skip] {scene} not in B2 pkl')
            continue

        b1_recs = b1_scenes[scene]
        b2_recs = b2_scenes[scene]
        scene_dir = os.path.join(args.out_dir, scene)
        os.makedirs(scene_dir, exist_ok=True)

        # Build a local_idx → record map for B2 (align with B1 frames)
        b2_by_idx = {r['local_idx']: r for r in b2_recs}

        # Per-frame comparison panels
        if not args.no_per_frame:
            print(f'  Rendering {len(b1_recs)} frames for {scene} ...')
            for rec_b1 in b1_recs:
                lidx = rec_b1['local_idx']
                rec_b2 = b2_by_idx.get(lidx)
                if rec_b2 is None:
                    continue
                cmp_path = os.path.join(scene_dir, f'b1_vs_b2_frame{lidx:03d}.png')
                render_frame_comparison(rec_b1, rec_b2, cmp_path, args.max_tracks, args.dpi)
                # Individual panels
                render_single_heatmap(rec_b1,
                    os.path.join(scene_dir, f'b1_frame{lidx:03d}.png'),
                    'B1 (gate-open)', args.max_tracks, args.dpi)
                render_single_heatmap(rec_b2,
                    os.path.join(scene_dir, f'b2_frame{lidx:03d}.png'),
                    'B2 (trained gate)', args.max_tracks, args.dpi)

        # Temporal trace
        trace_path = os.path.join(scene_dir, 'temporal_trace.png')
        render_temporal_trace(b1_recs, b2_recs, trace_path, scene, args.dpi)
        print(f'  Saved temporal trace: {trace_path}')

    print('[vis_alpha_heatmap] Done.')


if __name__ == '__main__':
    main()
