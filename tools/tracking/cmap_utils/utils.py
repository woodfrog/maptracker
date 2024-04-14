import cv2
from PIL import Image, ImageDraw
import os
import torch
import numpy as np
from shapely.geometry import LineString

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
    canvas = np.zeros((cfg.canvas_size[1], cfg.canvas_size[0]))
    for label, vecs in vectors.items():
        if label == 0:
            masks[label] = draw_polygons(vecs, roi_size, origin, cfg)
        else:
            masks[label] = draw_polylines(vecs, roi_size, origin, cfg)
        for mask in masks[label]:
            canvas += mask
    return masks, canvas


def interp_fixed_num(vector, num_pts):
    line = LineString(vector)

    distances = np.linspace(0, line.length, num_pts)
    sampled_points = np.array([list(line.interpolate(distance).coords) 
        for distance in distances]).squeeze()
    
    return sampled_points

def chamfer_distance_batch(pred_lines, gt_lines):

    _, num_pts, coord_dims = pred_lines.shape
    
    if not isinstance(pred_lines, torch.Tensor):
        pred_lines = torch.tensor(pred_lines)
    if not isinstance(gt_lines, torch.Tensor):
        gt_lines = torch.tensor(gt_lines)
    dist_mat = torch.cdist(pred_lines.view(-1, coord_dims), 
                    gt_lines.view(-1, coord_dims), p=2) 
    # (num_query*num_points, num_gt*num_points)
    dist_mat = torch.stack(torch.split(dist_mat, num_pts)) 
    # (num_query, num_points, num_gt*num_points)
    dist_mat = torch.stack(torch.split(dist_mat, num_pts, dim=-1)) 
    # (num_gt, num_q, num_pts, num_pts)

    dist1 = dist_mat.min(-1)[0].sum(-1)
    dist2 = dist_mat.min(-2)[0].sum(-1)

    dist_matrix = (dist1 + dist2).transpose(0, 1) / (2 * num_pts)
    
    return dist_matrix.numpy()

def average_precision(recalls, precisions, mode='area'):

    recalls = recalls[np.newaxis, :]
    precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = 0.
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        
        ind = np.where(mrec[0, 1:] != mrec[0, :-1])[0]
        ap = np.sum(
            (mrec[0, ind + 1] - mrec[0, ind]) * mpre[0, ind + 1])
    
    elif mode == '11points':
        for thr in np.arange(0, 1 + 1e-3, 0.1):
            precs = precisions[0, recalls[i, :] >= thr]
            prec = precs.max() if precs.size > 0 else 0
            ap += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')

    return ap
