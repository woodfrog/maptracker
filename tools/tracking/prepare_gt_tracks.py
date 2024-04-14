import argparse
import mmcv
from mmcv import Config
import os
from mmdet3d.datasets import build_dataset, build_dataloader
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
import copy
import imageio
from scipy.optimize import linear_sum_assignment
import pickle
from functools import partial
from multiprocessing import Pool


font                   = cv2.FONT_HERSHEY_SIMPLEX
location               = (200,60)
fontScale              = 2
fontColor              = (255,0,0)
thickness              = 2
lineType               = 2

N_WORKERS = 16

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--result', 
        default=None,
        help='prediction result to visualize'
        'If submission file is not provided, only gt will be visualized')
    parser.add_argument(
        '--out-dir', 
        default='demo',
        help='directory where visualize results will be saved')
    parser.add_argument(
        '--visualize', 
        action="store_true",
        default=False,
        help='whether visualize the formed gt tracks')
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
                

def draw_polylines(vecs, roi_size, origin, cfg):
    results = []
    for line_coords in vecs:
        canvas = np.zeros((cfg.canvas_size[1], cfg.canvas_size[0]), dtype=np.uint8)
        coords = (line_coords - origin) / roi_size * torch.tensor(cfg.canvas_size)
        coords = coords.numpy()
        cv2.polylines(canvas, np.int32([coords]), False, color=1, thickness=cfg.thickness)
        result = np.flipud(canvas)
        if result.sum() < 20:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            result = cv2.dilate(result, kernel, iterations=1)
        results.append(result)
    return results
        

def draw_polygons(vecs, roi_size, origin, cfg):
    results = []
    for poly_coords in vecs:
        mask = Image.new("L", size=(cfg.canvas_size[0], cfg.canvas_size[1]), color=0)
        coords = (poly_coords - origin) / roi_size * torch.tensor(cfg.canvas_size)
        coords = coords.numpy()
        vert_list = [(x, y) for x, y in coords]
        if not (coords[0] == coords[-1]).all():
            vert_list.append(vert_list[0])
        ImageDraw.Draw(mask).polygon(vert_list, outline=1, fill=1)
        result = np.flipud(np.array(mask))
        if result.sum() < 20:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            result = cv2.dilate(result, kernel, iterations=1)
        results.append(result)
    return results
        

def draw_instance_masks(vectors, roi_size, origin, cfg):
    masks = {}
    for label, vecs in vectors.items():
        if label == 0:
            masks[label] = draw_polygons(vecs, roi_size, origin, cfg)
        else:
            masks[label] = draw_polylines(vecs, roi_size, origin, cfg)
    return masks


def _mask_iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union


def find_matchings(src_masks, tgt_masks, thresh=0.1):
    """Find the matching of map elements between two temporally 
    connected frame

    Args:
        src_masks (_type_): instance masks of prev frame
        tgt_masks (_type_): instance masks of current frame
        thresh (float, optional): IOU threshold for matching. Defaults to 0.1.
    """
    matchings = {}
    for label, src_instances in src_masks.items():
        tgt_instances = tgt_masks[label]
        cost = np.zeros([len(src_instances), len(tgt_instances)])
        for i, src_ins in enumerate(src_instances):
            for j, tgt_ins in enumerate(tgt_instances):
                iou = _mask_iou(src_ins, tgt_ins)
                cost[i, j] = -iou
        row_ind, col_ind = linear_sum_assignment(cost)
        
        label_matching = [-1 for _ in range(len(src_instances))]
        label_matching_reverse = [-1 for _ in range(len(tgt_instances))]
        for i, j in zip(row_ind, col_ind):
            if -cost[i, j] > thresh:
                label_matching[i] = j
                label_matching_reverse[j] = i
        
        matchings[label] = (label_matching, label_matching_reverse)
    return matchings
                
        
def match_two_consecutive_frames(prev_data, curr_data, roi_size, origin, cfg):
    # get relative pose
    prev_e2g_trans = torch.tensor(prev_data['img_metas'].data['ego2global_translation'], dtype=torch.float64)
    prev_e2g_rot = torch.tensor(prev_data['img_metas'].data['ego2global_rotation'], dtype=torch.float64)
    curr_e2g_trans  = torch.tensor(curr_data['img_metas'].data['ego2global_translation'], dtype=torch.float64)
    curr_e2g_rot = torch.tensor(curr_data['img_metas'].data['ego2global_rotation'], dtype=torch.float64)
    prev_e2g_matrix = torch.eye(4, dtype=torch.float64)
    prev_e2g_matrix[:3, :3] = prev_e2g_rot
    prev_e2g_matrix[:3, 3] = prev_e2g_trans

    curr_g2e_matrix = torch.eye(4, dtype=torch.float64)
    curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
    curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

    prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix

    # get vector data
    prev_vectors = copy.deepcopy(prev_data['vectors'].data)
    curr_vectors = copy.deepcopy(curr_data['vectors'].data)

    #meta_info = curr_data['img_metas'].data
    #imgs = [mmcv.imread(i) for i in meta_info['img_filenames']]
    #cam_extrinsics = meta_info['cam_extrinsics']
    #cam_intrinsics = meta_info['cam_intrinsics']
    #ego2cams = meta_info['ego2cam']
    
    # transform prev vectors
    prev2curr_vectors = dict()
    for label, vecs in prev_vectors.items():
        if len(vecs) > 0:
            vecs = np.stack(vecs, 0)
            vecs = torch.tensor(vecs)
            N, num_points, _ = vecs.shape
            denormed_vecs = vecs * roi_size + origin # (num_prop, num_pts, 2)
            denormed_vecs = torch.cat([
                denormed_vecs,
                denormed_vecs.new_zeros((N, num_points, 1)), # z-axis
                denormed_vecs.new_ones((N, num_points, 1)) # 4-th dim
            ], dim=-1) # (num_prop, num_pts, 4)

            transformed_vecs = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_vecs.double()).float()
            normed_vecs = (transformed_vecs[..., :2] - origin) / roi_size # (num_prop, num_pts, 2)
            normed_vecs = torch.clip(normed_vecs, min=0., max=1.)
            prev2curr_vectors[label] = normed_vecs
        else:
            prev2curr_vectors[label] = vecs
    
    # convert to ego space for visualization
    for label in prev2curr_vectors:
        if len(prev2curr_vectors[label]) > 0:
            prev2curr_vectors[label] = prev2curr_vectors[label] * roi_size + origin
        if len(curr_vectors[label]) > 0:
            curr_vecs = torch.tensor(np.stack(curr_vectors[label]))
            curr_vectors[label] = curr_vecs * roi_size + origin
    
    prev2curr_masks = draw_instance_masks(prev2curr_vectors, roi_size, origin, cfg)
    curr_masks = draw_instance_masks(curr_vectors, roi_size, origin, cfg)
    
    prev2curr_matchings = find_matchings(prev2curr_masks, curr_masks, thresh=0.01)

    # For viz purpose, may display the maps in perspective images
    #viz_dir = os.path.join(scene_dir, '{}_viz_perspective'.format(local_idx))
    #if not os.path.exists(viz_dir):
    #    os.makedirs(viz_dir)
    #renderer.render_camera_views_from_vectors(curr_vectors, imgs, 
    #            cam_extrinsics, cam_intrinsics, ego2cams, 2, viz_dir)

    #renderer.render_bev_from_vectors(curr_vectors, out_dir=None, specified_path='cur.png')
    #renderer.render_bev_from_vectors(prev2curr_vectors, out_dir=None, specified_path='prev2cur.png')
    #from PIL import Image 
    #background = Image.open("cur.png")
    #overlay = Image.open("prev2cur.png")
    #background = background.convert("RGBA")
    #overlay = overlay.convert("RGBA")
    #new_img = Image.blend(background, overlay, 0.5)
    #new_img.save("cur_overlapped.png","PNG")
    #import pdb; pdb.set_trace()
    
    return prev2curr_matchings


def assign_global_ids(matchings_seq, vectors_seq):
    ids_seq = []
    global_map_index = {
        0: 0,
        1: 0,
        2: 0,
    }
    
    ids_0 = dict()
    for label, vectors in vectors_seq[0].items():
        id_mapping = dict()
        for i, _ in enumerate(vectors):
            id_mapping[i] = global_map_index[label]
            global_map_index[label] += 1
        ids_0[label] = id_mapping
    ids_seq.append(ids_0)

    # Trace all frames following the consecutive matching
    for t, vectors_t in enumerate(vectors_seq[1:]):
        ids_t = dict()
        for label, vectors in vectors_t.items():
            reverse_matching = matchings_seq[t][label][1]
            id_mapping = dict()
            for i, _ in enumerate(vectors):
                if reverse_matching[i] != -1:
                    prev_id = reverse_matching[i]
                    global_id = ids_seq[-1][label][prev_id]
                else:
                    global_id = global_map_index[label]
                    global_map_index[label] += 1
                id_mapping[i] = global_id
            ids_t[label] = id_mapping
        ids_seq.append(ids_t)
    return ids_seq


def _denorm(vectors, roi_size, origin):
    for label in vectors:
        for i, vec in enumerate(vectors[label]):
            vectors[label][i] = vec * roi_size + origin
    return vectors


def form_gt_track_single(scene_name, scene_name2idx, dataset, out_dir, cfg, args):
    print('Process scene {}'.format(scene_name))

    renderer = dataset.renderer

    roi_size = torch.tensor(cfg.roi_size)
    origin = torch.tensor(cfg.pc_range[:2])

    start_idx = scene_name2idx[scene_name][0]
    matchings_seq = []
    vectors_seq = []

    for idx in scene_name2idx[scene_name]:
        local_idx = idx - start_idx
        if idx == start_idx:
            prev_data = dataset[idx]
        if idx == scene_name2idx[scene_name][-1]: # prev_data is the last frame
            vectors_seq.append(prev_data['vectors'].data)
            break

        curr_data = dataset[idx+1]
        matchings = match_two_consecutive_frames(prev_data, curr_data, roi_size, origin, cfg)
        matchings_seq.append(matchings)
        vectors_seq.append(prev_data['vectors'].data)

        prev_data = curr_data
    
    # Derive global ids...
    # get global ids by traversing all consecutive matching results
    ids_info = assign_global_ids(matchings_seq, vectors_seq)

    matching_meta = {
        'sample_ids':scene_name2idx[scene_name],
        'instance_ids': ids_info,
    }

    if args.visualize:
        print('Visualize gt tracks for scene {}'.format(scene_name))
        scene_dir = os.path.join(out_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        # visualize with matched track ids
        imgs = []
        for idx, (id_info, vectors) in enumerate(zip(ids_info, vectors_seq)):
            vectors = _denorm(vectors, roi_size.numpy(), origin.numpy())
            save_path = os.path.join(scene_dir, f'{idx}_with_id.png')
            renderer.render_bev_from_vectors(vectors, out_dir=None, specified_path=save_path, id_info=id_info)
            viz_img = np.ascontiguousarray(cv2.imread(save_path)[:, :, ::-1], dtype=np.uint8)
            if idx == 0:
                img_shape = (viz_img.shape[1], viz_img.shape[0])
            else:
                viz_img = cv2.resize(viz_img, img_shape)
            cv2.putText(viz_img, 't={}'.format(idx), location, font, fontScale, fontColor,
            thickness, lineType)
            imgs.append(viz_img)
        gif_path = os.path.join(scene_dir, 'matching.gif')
        imageio.mimsave(gif_path, imgs, duration=500)
    
    return scene_name, matching_meta
        
        
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)

    for split in ['train', 'val']:
        if split == 'train' and split not in cfg.match_config.ann_file:
            cfg.match_config.ann_file = cfg.match_config.ann_file.replace('val', 'train')
        if split == 'val' and split not in cfg.match_config.ann_file:
            cfg.match_config.ann_file = cfg.match_config.ann_file.replace('train', 'val')

        # build the dataset
        dataset = build_dataset(cfg.match_config)

        scene_name2idx = {}
        for idx, sample in enumerate(dataset.samples):
            scene = sample['scene_name']
            if scene not in scene_name2idx:
                scene_name2idx[scene] = []
            scene_name2idx[scene].append(idx)
            
        all_scene_names = sorted(list(scene_name2idx.keys()))
        all_scene_matching_meta = {}

        out_dir = os.path.join(args.out_dir, split)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        all_scene_infos = []
        for scene_idx, scene_name in enumerate(all_scene_names):
            all_scene_infos.append((scene_name,))
            
        if N_WORKERS > 0:
            fn = partial(form_gt_track_single, scene_name2idx=scene_name2idx,
                dataset=dataset, cfg=cfg, out_dir=out_dir, args=args)
            pool = Pool(N_WORKERS)
            matching_results = pool.starmap(fn, all_scene_infos)
            pool.close()
        else:
            matching_results =[]
            for scene_info in all_scene_infos:
                scene_name = scene_info[0]
                single_matching_result = form_gt_track_single(scene_name=scene_name, scene_name2idx=scene_name2idx,
                        dataset=dataset, cfg=cfg, out_dir=out_dir, args=args)
                matching_results.append(single_matching_result)
        
        for scene_name, matching_meta in matching_results:
            all_scene_matching_meta[scene_name] = matching_meta
        
        track_gt_path = cfg.match_config.ann_file[:-4] + '_gt_tracks.pkl'
        with open(track_gt_path, 'wb') as f:
            pickle.dump(all_scene_matching_meta, f, protocol=pickle.HIGHEST_PROTOCOL)

        
if __name__ == '__main__':
    main()
