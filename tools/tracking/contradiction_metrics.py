import argparse
import json
import os
import os.path as osp

import mmcv
import numpy as np
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute contradiction-suite metrics from a condition eval output.')
    parser.add_argument('config', help='config file path')
    parser.add_argument('submission_path', help='path to submission_vector.json')
    parser.add_argument('--onset', type=int, required=True, help='fixed contradiction onset frame index')
    parser.add_argument('--score-thr', type=float, default=0.4, help='prediction score threshold')
    parser.add_argument('--match-thr', type=float, default=1.5,
                        help='chamfer distance threshold for gt matching when counting stale FP')
    parser.add_argument('--mode', choices=['c_full', 'c_tail', 'clean'], required=True)
    parser.add_argument('--stale-offset', type=int, required=True)
    parser.add_argument('--out-path', default=None, help='metrics json output path')
    return parser.parse_args()


def import_plugin(cfg):
    import importlib
    if hasattr(cfg, 'plugin') and cfg.plugin:
        if hasattr(cfg, 'plugin_dir'):
            plugin_dirs = cfg.plugin_dir
            if not isinstance(plugin_dirs, list):
                plugin_dirs = [plugin_dirs]
            for plugin_dir in plugin_dirs:
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                importlib.import_module(_module_path)


def polyline_length(vec):
    pts = np.asarray(vec).reshape(-1, 2)
    if pts.shape[0] <= 1:
        return 0.0
    diffs = pts[1:] - pts[:-1]
    return float(np.linalg.norm(diffs, axis=1).sum())


def chamfer_distance(a, b):
    a = np.asarray(a).reshape(-1, 2)
    b = np.asarray(b).reshape(-1, 2)
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(dist.min(axis=1).mean() + dist.min(axis=0).mean()) / 2.0


def build_gt_by_token(gts_norm, roi_size, origin):
    gt = {}
    for token, vectors in gts_norm.items():
        denorm_vectors = {}
        for label, vecs in vectors.items():
            denorm_vectors[int(label)] = []
            for vec in vecs:
                vec_np = np.asarray(vec)
                denorm_vectors[int(label)].append(vec_np * roi_size + origin)
        gt[token] = denorm_vectors
    return gt


def build_scene_frames(dataset):
    scene_frames = {}
    for sample in dataset.samples:
        token = sample['token']
        scene = sample['scene_name']
        local_idx = int(sample.get('sample_idx', sample.get('modified_sample_idx', 0)))
        scene_frames.setdefault(scene, []).append((local_idx, token))

    for scene in scene_frames:
        scene_frames[scene] = [t for _, t in sorted(scene_frames[scene], key=lambda x: x[0])]
    return scene_frames


def pred_is_stale_fp(pred_vec, pred_label, gt_vectors, match_thr):
    gt_candidates = gt_vectors.get(pred_label, [])
    if len(gt_candidates) == 0:
        return True
    for gt_vec in gt_candidates:
        if chamfer_distance(pred_vec, gt_vec) <= match_thr:
            return False
    return True


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    from plugin.datasets.evaluation.vector_eval import VectorEvaluate
    evaluator = VectorEvaluate(cfg.data.test)
    dataset = evaluator.dataset

    submission = mmcv.load(args.submission_path)
    pred_results = submission['results']
    roi_size = np.asarray(cfg.roi_size, dtype=np.float32)
    # Align with BaseMapDataset.format_results denormalization:
    # vector = vector * roi_size + (-roi_size / 2)
    origin = -0.5 * roi_size
    gt_by_token = build_gt_by_token(evaluator.gts, roi_size, origin)
    scene_frames = build_scene_frames(dataset)

    scene_stale_lengths = {}
    scene_persistence = {}
    scene_qual = {}

    for scene, tokens in scene_frames.items():
        stale_by_frame = []
        for frame_idx, token in enumerate(tokens):
            if token not in pred_results:
                stale_by_frame.append(0.0)
                continue
            preds = pred_results[token]
            gt_vectors = gt_by_token[token]
            stale_len = 0.0
            for vec, label, score in zip(preds['vectors'], preds['labels'], preds['scores']):
                if float(score) < args.score_thr:
                    continue
                if pred_is_stale_fp(vec, int(label), gt_vectors, args.match_thr):
                    stale_len += polyline_length(vec)
            stale_by_frame.append(stale_len)

        post = stale_by_frame[args.onset:] if args.onset < len(stale_by_frame) else []
        persistence = 0
        for val in post:
            if val > 0:
                persistence += 1
            else:
                break

        scene_stale_lengths[scene] = stale_by_frame
        scene_persistence[scene] = persistence
        scene_qual[scene] = {
            'onset': args.onset,
            'stale_false_positive_length_per_frame': stale_by_frame,
        }

    all_post_vals = []
    for vals in scene_stale_lengths.values():
        all_post_vals.extend(vals[args.onset:] if args.onset < len(vals) else [])

    mean_persistence = float(np.mean(list(scene_persistence.values()))) if scene_persistence else 0.0
    cumulative_stale_fp_length = float(np.sum(all_post_vals)) if all_post_vals else 0.0

    alpha_separation = None
    alpha_path = osp.join(osp.dirname(args.submission_path), 'alpha_stats.json')
    if osp.exists(alpha_path):
        with open(alpha_path, 'r') as f:
            alpha_stats = json.load(f)
        if 'alpha_mean_preserved_recent' in alpha_stats and 'alpha_mean_affected' in alpha_stats:
            alpha_separation = float(alpha_stats['alpha_mean_preserved_recent']) - float(alpha_stats['alpha_mean_affected'])

    out = {
        'mode': args.mode,
        'stale_offset': int(args.stale_offset),
        'onset': int(args.onset),
        'metrics': {
            'stale_persistence_time_mean': mean_persistence,
            'cumulative_stale_false_positive_polyline_length': cumulative_stale_fp_length,
            'stale_false_positive_polyline_length_is_proxy': True,
            'alpha_separation_summary': alpha_separation,
        },
        'scene_metrics': {
            'persistence_time_by_scene': scene_persistence,
        },
        'scene_qualitative': scene_qual,
    }

    out_path = args.out_path
    if out_path is None:
        out_path = osp.join(osp.dirname(args.submission_path), 'contradiction_metrics.json')
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[contradiction_metrics] wrote: {out_path}')


if __name__ == '__main__':
    main()
