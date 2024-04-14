import argparse
import mmcv
from mmcv import Config
import os
from mmdet3d.datasets import build_dataset, build_dataloader
import imageio
import cv2
import gc

font                   = cv2.FONT_HERSHEY_SIMPLEX
location               = (200,60)
fontScale              = 2
fontColor              = (255,0,0)
thickness              = 2
lineType               = 2

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('idx', type=int,
        help='which scene to visualize')
    parser.add_argument('--result', 
        default=None,
        help='prediction result to visualize'
        'If submission file is not provided, only gt will be visualized')
    parser.add_argument('--thr', 
        type=float,
        default=0.4,
        help='score threshold to filter predictions')
    parser.add_argument(
        '--out-dir', 
        default='demo',
        help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args

def import_plugin(cfg):
    '''
        import modules from plguin/xx, registry will be update
    '''

    import sys
    sys.path.append(os.path.abspath('.'))    
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            
            def import_path(plugin_dir):
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            plugin_dirs = cfg.plugin_dir
            if not isinstance(plugin_dirs, list):
                plugin_dirs = [plugin_dirs,]
            for plugin_dir in plugin_dirs:
                import_path(plugin_dir)

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)

    from plugin.datasets.evaluation.vector_eval import VectorEvaluate

    # build the dataset
    dataset = build_dataset(cfg.eval_config)

    # ann_file = mmcv.load('datasets/nuScenes/nuscenes_map_infos_val.pkl')
    scene_name2idx = {}
    for idx, sample in enumerate(dataset.samples):
        scene = sample['scene_name']
        if scene not in scene_name2idx:
            scene_name2idx[scene] = []
        scene_name2idx[scene].append(idx)

    results = mmcv.load(args.result)

    for scene_idx in range(args.idx):
        scene_name = sorted(list(scene_name2idx.keys()))[scene_idx]
        print(scene_name)
        scene_dir = os.path.join(args.out_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        start_idx = scene_name2idx[scene_name][0]

        gt_imgs = []
        pred_imgs = []
        pred_masks = []
        pred_score_imgs = []
        local_idx = 0
        for idx in mmcv.track_iter_progress(scene_name2idx[scene_name]):
            out_dir = os.path.join(scene_dir, str(idx - start_idx + 1))
            gt_dir = os.path.join(out_dir, 'gt')
            pred_dir = os.path.join(out_dir, 'pred')
            pred_score_dir = os.path.join(out_dir, 'pred_score')
            track_dir = os.path.join(out_dir, 'track_input')

            if args.result is not None:
                os.makedirs(pred_dir, exist_ok=True)
                os.makedirs(pred_score_dir, exist_ok=True)
                os.makedirs(track_dir, exist_ok=True)
                dataset.show_result(
                        submission=results, 
                        idx=idx, 
                        score_thr=args.thr, 
                        out_dir=pred_dir,
                        draw_score=False,
                        show_semantic=True,
                    )
                dataset.show_result(
                        submission=results, 
                        idx=idx, 
                        score_thr=args.thr, 
                        out_dir=pred_score_dir,
                        draw_score=True,
                    )
                dataset.show_track(
                        submission=results, 
                        idx=idx, 
                        out_dir=track_dir,
                    )
            os.makedirs(gt_dir, exist_ok=True)
            dataset.show_gt(idx, gt_dir)
            
            # Generate GIF animation
            pred_path = os.path.join(pred_dir, 'map.jpg')
            pred_image = imageio.imread(pred_path)
            pred_score_path = os.path.join(pred_score_dir, 'map.jpg')
            pred_score_image = imageio.imread(pred_score_path)
            pred_mask_path = os.path.join(pred_dir, 'semantic_map.jpg')
            pred_mask_image = imageio.imread(pred_mask_path)

            if local_idx == 0:
                pred_size = (pred_score_image.shape[1], pred_score_image.shape[0])
            else:
                pred_score_image = cv2.resize(pred_score_image, pred_size)

            gt_path = os.path.join(gt_dir, 'map.jpg')
            gt_image = imageio.imread(gt_path)
            cv2.putText(gt_image,'t={}'.format(local_idx), location, font, fontScale, fontColor,
                thickness, lineType)
            cv2.putText(pred_image,'t={}'.format(local_idx), location, font, fontScale, fontColor,
                thickness, lineType)
            cv2.putText(pred_score_image,'t={}'.format(local_idx), location, font, fontScale, fontColor,
                thickness, lineType)
            cv2.putText(pred_mask_image,'t={}'.format(local_idx), (50, 15), font, 0.5, (210, 52, 235),
                thickness, lineType)

            pred_imgs.append(pred_image)
            pred_score_imgs.append(pred_score_image)
            pred_masks.append(pred_mask_image)
            gt_imgs.append(gt_image)

            local_idx += 1
        
        gt_gif_path = os.path.join(scene_dir, 'gt_seq.gif')
        pred_gif_path = os.path.join(scene_dir, 'pred_seq.gif')
        pred_mask_gif_path = os.path.join(scene_dir, 'pred_mask_seq.gif')
        pred_score_gif_path = os.path.join(scene_dir, 'pred_score_seq.gif')
        imageio.mimsave(gt_gif_path, gt_imgs)
        imageio.mimsave(pred_gif_path, pred_imgs)
        imageio.mimsave(pred_mask_gif_path, pred_masks, duration=350)
        imageio.mimsave(pred_score_gif_path, pred_score_imgs, duration=250)
        gc.collect() # some versions of matplotlib has memory leak issue


if __name__ == '__main__':
    main()
