import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from .utils import *

cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}

def get_prev2curr_matrix(prev_meta,curr_meta):
    # get relative pose
    prev_e2g_trans = torch.tensor(prev_meta['ego2global_translation'], dtype=torch.float64)
    prev_e2g_rot = torch.tensor(prev_meta['ego2global_rotation'], dtype=torch.float64)
    curr_e2g_trans = torch.tensor(curr_meta['ego2global_translation'], dtype=torch.float64)
    curr_e2g_rot = torch.tensor(curr_meta['ego2global_rotation'], dtype=torch.float64)
    
    prev_e2g_matrix = torch.eye(4, dtype=torch.float64)
    prev_e2g_matrix[:3, :3] = prev_e2g_rot
    prev_e2g_matrix[:3, 3] = prev_e2g_trans

    curr_g2e_matrix = torch.eye(4, dtype=torch.float64)
    curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
    curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

    prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
    return prev2curr_matrix


def find_matchings_iou(src_masks, tgt_masks, thresh=0.1):
    """Find the matching of map elements between two temporally 
    connected frame

    Args:
        src_masks (_type_): instance masks of prev frame
        tgt_masks (_type_): instance masks of current frame
        thresh (float, optional): IOU threshold for matching. Defaults to 0.1.
    """
    def _mask_iou(mask1, mask2):
        intersection = (mask1 * mask2).sum()
        if intersection == 0:
            return 0.0
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union
    
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

def find_matchings_chamfer(pred_vectors, gt_vectors, score_dict,thresh=0.5):
    matchings = {}
    for label, src_instances in pred_vectors.items():
        tgt_instances = gt_vectors[label]
        num_gts = len(tgt_instances)
        num_preds = len(src_instances)
        label_matching = [-1 for _ in range(len(src_instances))]
        label_matching_reverse = [-1 for _ in range(len(tgt_instances))]
        if len(src_instances) == 0 or len(tgt_instances)==0:
            matchings[label] = (label_matching, label_matching_reverse)
            continue
        cdist = chamfer_distance_batch(src_instances, tgt_instances)
        label_score = np.array(score_dict[label])
        matrix_min = cdist.min(axis=1)

        # for each det, which gt is the closest to it
        matrix_argmin = cdist.argmin(axis=1)
        sort_inds = np.argsort(-label_score)
        gt_covered = np.zeros(num_gts, dtype=bool)

        tp = np.zeros((num_preds), dtype=np.float32)
        fp = np.zeros((num_preds), dtype=np.float32)
        for i in sort_inds:
            if matrix_min[i] <= thresh:
                matched_gt = matrix_argmin[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    label_matching[i] = matched_gt
                    label_matching_reverse[matched_gt] = i
        matchings[label] = (label_matching, label_matching_reverse)
    return matchings

def get_consecutive_vectors(prev_vectors,curr_vectors,prev2curr_matrix,origin,roi_size):
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
        if len(prev_vectors[label]) > 0:
            prev_vecs = torch.tensor(np.stack(prev_vectors[label]))
            prev_vectors[label] = prev_vecs * roi_size + origin
    
    return prev_vectors, curr_vectors, prev2curr_vectors

def filter_vectors(data_info, origin,roi_size,thr,num_interp=20):
    ### filter vectors over threshold
    filtered_vectors = {label: [] for label in cat2id.values()}
    for i in range(len(data_info['labels'])):
        score = data_info['scores'][i]
        label = data_info['labels'][i]
        v = data_info['vectors'][i]
        if score > thr:
            interp_v = interp_fixed_num(v,num_interp)
            filtered_vectors[label].append( (np.array(interp_v) - origin)/roi_size )
    return filtered_vectors
