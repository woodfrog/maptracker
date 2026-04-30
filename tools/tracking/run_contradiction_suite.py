import argparse
import itertools
import json
import os
import os.path as osp
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run contradiction-suite eval matrix for MapTracker temporal-gating MVP.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--work-root', required=True,
                        help='root directory for condition-tagged outputs')
    parser.add_argument('--modes', nargs='+', default=['c_full', 'c_tail'],
                        choices=['c_full', 'c_tail'], help='contradiction corruption modes')
    parser.add_argument('--stale-offsets', nargs='+', type=int, default=[4, 8],
                        help='list of stale offsets for suite matrix')
    parser.add_argument('--onset', type=int, required=True, help='fixed corruption onset for all suite runs')
    parser.add_argument('--c-tail-keep-recent', type=int, default=1)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--gpus', type=int, default=1, help='used only when launcher=pytorch')
    parser.add_argument('--port', type=int, default=29500, help='used only when launcher=pytorch')
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--extra-cfg-options', nargs='+', default=None,
                        help='extra cfg overrides passed through to tools/test.py')
    parser.add_argument('--dry-run', action='store_true',
                        help='print planned commands without executing')
    parser.add_argument('--skip-summary', action='store_true',
                        help='run inference and per-condition metrics but skip writing '
                             'contradiction_suite_summary.json; use when conditions run '
                             'in parallel and a dedicated aggregation job writes the summary')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='skip inference and per-condition metrics; load existing '
                             'contradiction_metrics.json from each condition dir and write '
                             'the final contradiction_suite_summary.json only')
    return parser.parse_args()


def build_test_command(args, mode, offset, cond_workdir):
    condition_tag = f'{mode}_offset{offset}_onset{args.onset}'
    cmd = [
        sys.executable, 'tools/test.py',
        args.config,
        args.checkpoint,
        '--work-dir', cond_workdir,
        '--eval',
        '--launcher', args.launcher,
        '--memory-corruption-mode', mode,
        '--memory-stale-offset', str(offset),
        '--memory-c-tail-keep-recent', str(args.c_tail_keep_recent),
        '--memory-corruption-onset', str(args.onset),
        '--condition-tag', condition_tag,
    ]
    if args.launcher == 'pytorch':
        cmd.extend(['--local_rank', '0'])
    if args.extra_cfg_options:
        cmd.append('--cfg-options')
        cmd.extend(args.extra_cfg_options)
    return cmd, condition_tag


def run_cmd(cmd, dry_run=False):
    print('[run]', ' '.join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def choose_qual_scene(condition_metrics):
    scenes = condition_metrics.get('scene_metrics', {}).get('persistence_time_by_scene', {})
    if not scenes:
        return None
    ranked = sorted(scenes.items(), key=lambda x: x[1], reverse=True)
    return ranked[0][0]


def main():
    args = parse_args()
    os.makedirs(args.work_root, exist_ok=True)

    matrix = list(itertools.product(args.modes, args.stale_offsets))
    aggregate = {
        'onset': args.onset,
        'conditions': {},
        'qualitative_examples': {},
    }

    for mode, offset in matrix:
        cond_name = f'{mode}_offset{offset}_onset{args.onset}'
        cond_workdir = osp.join(args.work_root, cond_name)
        os.makedirs(cond_workdir, exist_ok=True)

        if args.aggregate_only:
            metrics_path = osp.join(cond_workdir, 'contradiction_metrics.json')
            if not osp.exists(metrics_path):
                raise FileNotFoundError(
                    f'[aggregate-only] metrics file not found: {metrics_path}\n'
                    f'  Run the corruption eval first for condition: {cond_name}')
            with open(metrics_path) as f:
                aggregate['conditions'][cond_name] = json.load(f)
            print(f'[aggregate-only] loaded: {metrics_path}')
            continue

        cmd, condition_tag = build_test_command(args, mode, offset, cond_workdir)
        ret = run_cmd(cmd, args.dry_run)
        if ret != 0:
            raise RuntimeError(f'Condition run failed: {condition_tag}')

        submission_path = osp.join(cond_workdir, 'submission_vector.json')
        metrics_path = osp.join(cond_workdir, 'contradiction_metrics.json')
        metrics_cmd = [
            sys.executable, 'tools/tracking/contradiction_metrics.py',
            args.config,
            submission_path,
            '--onset', str(args.onset),
            '--mode', mode,
            '--stale-offset', str(offset),
            '--out-path', metrics_path,
        ]
        ret = run_cmd(metrics_cmd, args.dry_run)
        if ret != 0:
            raise RuntimeError(f'Metric computation failed: {condition_tag}')

        if args.dry_run:
            continue

        with open(metrics_path, 'r') as f:
            cond_metrics = json.load(f)
        aggregate['conditions'][cond_name] = cond_metrics

    if not args.dry_run and not args.skip_summary:
        for required_mode in ['c_full', 'c_tail']:
            mode_conds = {
                k: v for k, v in aggregate['conditions'].items()
                if v.get('mode') == required_mode
            }
            if mode_conds:
                best_cond = sorted(
                    mode_conds.items(),
                    key=lambda kv: kv[1]['metrics']['stale_persistence_time_mean'],
                    reverse=True,
                )[0]
                qual_scene = choose_qual_scene(best_cond[1])
                aggregate['qualitative_examples'][required_mode] = {
                    'condition': best_cond[0],
                    'scene': qual_scene,
                    'scene_qualitative_path': osp.join(
                        args.work_root, best_cond[0], 'contradiction_metrics.json'),
                }

        out_path = osp.join(args.work_root, 'contradiction_suite_summary.json')
        with open(out_path, 'w') as f:
            json.dump(aggregate, f, indent=2)
        print(f'[suite] wrote summary: {out_path}')


if __name__ == '__main__':
    main()
