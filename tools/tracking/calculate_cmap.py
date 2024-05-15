import argparse
from mmcv import Config
from mmdet3d.datasets import build_dataset
import cv2
import torch
import numpy as np
import pickle
import time

from cmap_utils.utils import *
from cmap_utils.match_utils import *
from cmap_utils.data_utils import *

font                   = cv2.FONT_HERSHEY_SIMPLEX
location               = (200,60)
fontScale              = 2
fontColor              = (255,0,0)
thickness              = 2
lineType               = 2

cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}

id2cat = {
    0:'ped_crossing',
    1:'divider',
    2:'boundary',
}

COLOR_MAPS_BGR = {
    # bgr colors
    'divider': (0, 0, 255),
    'boundary': (0, 255, 0),
    'ped_crossing': (255, 0, 0),
    'centerline': (51, 183, 255),
    'drivable_area': (171, 255, 255)
}

COLOR_MAPS_PLT = {
    'divider': 'r',
    'boundary': 'g',
    'ped_crossing': 'b',
    'centerline': 'orange',
    'drivable_area': 'y',
}

INTERP_NUM = 200
N_WORKERS = 0

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--thr', 
        type=float,
        default=0.4,
        help='score threshold to filter predictions')
    parser.add_argument(
        '--result_path',
        default=None,
        help='directory to submission file')
    parser.add_argument(
        '--consist',
        default=1,
        type=int,
        help='whether to use the consistent criterion'
    )
    parser.add_argument(
        '--cons_frames',
        default=5,
        help='consective frames for cons metric'
    )
    args = parser.parse_args()
    return args

def instance_match(pred_lines, scores, gt_lines, threshold, metric='chamfer'):
    ### obtain tp,fp,score for a frame based on chamfer distance

    num_preds = pred_lines.shape[0]
    num_gts = gt_lines.shape[0]

    # tp and fp
    tp = np.zeros((num_preds), dtype=np.float32)
    fp = np.zeros((num_preds), dtype=np.float32)

    if num_gts == 0:
        fp[...] = 1
        return (tp.copy(),fp.copy())
    
    if num_preds == 0:
        return (tp.copy(),fp.copy())

    assert pred_lines.shape[1] == gt_lines.shape[1], \
        "sample points num should be the same"

    matrix = np.zeros((num_preds, num_gts))
    matrix = chamfer_distance_batch(pred_lines, gt_lines)
    matrix_min = matrix.min(axis=1)
    matrix_argmin = matrix.argmin(axis=1)
    sort_inds = np.argsort(-scores)
    tp = np.zeros((num_preds), dtype=np.float32)
    fp = np.zeros((num_preds), dtype=np.float32)
    gt_covered = np.zeros(num_gts, dtype=bool)
    for i in sort_inds:
        if matrix_min[i] <= threshold:
            matched_gt = matrix_argmin[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    return (tp.copy(),fp.copy())

def _evaluate_single(pred_vectors, scores, gt_vectors, threshold, metric='chamfer'):
    ### collect tp-fp-score information

    pred_lines = np.array(pred_vectors)
    gt_lines = np.array(gt_vectors)
    
    if len(pred_lines) == 0 or len(gt_lines)==0:
        tp_fp_score = np.zeros((0,3))
        return tp_fp_score
    scores = np.array(scores)
    tp_fp_list = instance_match(pred_lines, scores, gt_lines, threshold, metric) # (M, 2)

    tp, fp = tp_fp_list
    tp_fp_score = np.hstack([tp[:, None], fp[:, None], scores[:, None]])
    return tp_fp_score

def match_gt_w_pred(curr_data,curr_data_gt,thresh):
    ### find local id matching between predicted vector and gt vectors

    curr_vectors_np = {label: [] for label in cat2id.values()}
    curr_scores_np = {label: [] for label in cat2id.values()}
    for i in range(len(curr_data['labels'])):
        score = curr_data['scores'][i]
        label = curr_data['labels'][i]
        v = curr_data['vectors'][i]
        curr_vectors_np[label].append(v)
        curr_scores_np[label].append(score)
    curr_vectors = {}
    for label, vecs in curr_vectors_np.items():
        if len(vecs) > 0:
            vecs = np.stack(vecs, 0)
            vecs = torch.tensor(vecs)
            curr_vectors[label] = vecs
        else:
            curr_vectors[label] = vecs
    curr_vectors_gt_np = curr_data_gt
    curr_vectors_gt = {}
    for label, vecs in curr_vectors_gt_np.items():
        if len(vecs) > 0:
            vecs_np = []
            for vec in vecs:
                vecs_np.append(vec)
            vecs = np.stack(vecs_np, 0)
            vecs = torch.tensor(vecs)
            curr_vectors_gt[label] = vecs
        else:
            curr_vectors_gt[label] = vecs
    pred2gt_matchings = find_matchings_chamfer(curr_vectors,curr_vectors_gt,curr_scores_np,thresh=thresh)

    return pred2gt_matchings

def get_scene_matching_result(gts,pred_results,scene_name2token,scene_name,thresh=1.5):
    ### obtain local id matching of a scene 

    start_token = scene_name2token[scene_name][0]
    vectors_seq = []
    scores_seq = []
    pred_matching_seq = []
    vectors_gt_seq = []
    pred2gt_matchings_seq = []

    choose_scene = pred_results[start_token]['scene_name']
    for local_idx,token in enumerate(scene_name2token[scene_name]):
        prev_data = pred_results[token]
        gt_vectors = gts[token]

        assert prev_data['scene_name']  == choose_scene
        assert prev_data['local_idx'] == local_idx

        vectors_gt_seq.append(gt_vectors)

        vectors = {label: [] for label in cat2id.values()}
        scores = {label: [] for label in cat2id.values()}
        pred_matching = {label: [] for label in cat2id.values()}
        for i in range(len(prev_data['labels'])):
            score, label, v,pred_glb_id = \
                prev_data['scores'][i], prev_data['labels'][i], prev_data['vectors'][i], prev_data['global_ids'][i]
            vectors[label].append(v)
            scores[label].append(score)
            pred_matching[label].append(pred_glb_id)
        pred_matching_seq.append(pred_matching)
        vectors_seq.append(vectors)
        scores_seq.append(scores)
        pred2gt_matchings = match_gt_w_pred(prev_data,gt_vectors, thresh)
        pred2gt_matchings_seq.append(pred2gt_matchings)

    return vectors_seq, pred_matching_seq, pred2gt_matchings_seq

def pred2gt_global_matching(ids_info,ids_info_gt,pred2gt_seq):
    ### obtain global id matching between predicted vectors and gt vectors of a scene

    pred2gt_global_seq = []
    for frame_idx in range(len(pred2gt_seq)):
        f_match = pred2gt_seq[frame_idx]
        f_ids_info = ids_info[frame_idx]
        f_ids_info_gt = ids_info_gt[frame_idx]
        pred2gt_match_dict = {}
        for label in f_ids_info.keys():
            pred2gt_match_dict[label] = {}
            f_label_match = f_match[label][0]
            f_ids_label_info,f_ids_label_info_gt = f_ids_info[label],f_ids_info_gt[label]
            for pred_match_idx, gt_match_idx in enumerate(f_label_match):
                pred_glb_match_idx = f_ids_label_info[pred_match_idx]

                if gt_match_idx != -1:
                    gt_glb_match_idx = f_ids_label_info_gt[gt_match_idx]
                else:
                    gt_glb_match_idx = -1
                pred2gt_match_dict[label][pred_glb_match_idx] = gt_glb_match_idx
        pred2gt_global_seq.append(pred2gt_match_dict)

    return pred2gt_global_seq

def get_tpfp_from_scene_single(scene_name,args,scene_name2token,pred_results,gts,
        gt_matching,threshold):
    
    ### generate tp-fp list in a single scene
    tpfp_score_record = {0:[],1:[],2:[]}
    scene_gt_matching = gt_matching[scene_name]['instance_ids']

    if args.consist:
        vectors_seq, scene_pred_matching,pred2gt_seq \
            = get_scene_matching_result(gts,pred_results,scene_name2token,scene_name,threshold)
        pred2gt_global_seq = pred2gt_global_matching(scene_pred_matching,scene_gt_matching,pred2gt_seq)

    vectors_seq = []
    scores_seq = []
    gt_flag_dict = {label:{} for label in cat2id.values()}
    for frame_idx, token in enumerate(scene_name2token[scene_name]):
        prev_data = pred_results[token]
        vectors_gt = gts[token]

        vectors = {label: [] for label in cat2id.values()}
        scores = {label: [] for label in cat2id.values()}
        for i in range(len(prev_data['labels'])):
            score, label, v = prev_data['scores'][i], prev_data['labels'][i], prev_data['vectors'][i]
            vectors[label].append(v)
            scores[label].append(score)
        
        for label in cat2id.values():
            tpfp_score = _evaluate_single(vectors[label], scores[label], vectors_gt[label] ,threshold)
            if args.consist:
                #### deal with the consistency part
                for vec_idx,single_tpfp_score in enumerate(tpfp_score):
                    curr_pred2gt_match = pred2gt_global_seq[frame_idx][label]  ### pred_global_id: gt_global_id

                    pred_local2global_mapping = scene_pred_matching[frame_idx][label]
                    match_glb_pred_idx = pred_local2global_mapping[vec_idx]    ### 
                    match_glb_gt_idx = curr_pred2gt_match[match_glb_pred_idx]

                    if match_glb_gt_idx not in gt_flag_dict[label].keys():
                        gt_flag_dict[label][match_glb_gt_idx] = match_glb_pred_idx
                    else:
                        if match_glb_pred_idx != gt_flag_dict[label][match_glb_gt_idx]:
                            tpfp_score[vec_idx][:2] = np.array([0,1])
            tpfp_score_record[label].append(tpfp_score)

        vectors_seq.append(vectors)
        scores_seq.append(scores)

    return tpfp_score_record

def get_mAP(tpfp_score_record,num_gts,threshold):

    ### calculate mean AP given tp-fp-score record
    result_dict = {}
    for cat_name,label in cat2id.items():
        sum_AP = 0
        result_dict[cat_name] = {}
        tp_fp_score = [np.vstack(i[label]) for i in tpfp_score_record]
        tp_fp_score = np.vstack(tp_fp_score)

        sort_inds = np.argsort(-tp_fp_score[:, -1])

        tp = tp_fp_score[sort_inds, 0]
        fp = tp_fp_score[sort_inds, 1]
        tp = np.cumsum(tp, axis=0)
        fp = np.cumsum(fp, axis=0)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[label], eps)
        precisions = tp/np.maximum(tp+fp, eps)

        AP = average_precision(recalls, precisions, 'area')
        sum_AP += AP
        result_dict[cat_name].update({f'AP@{threshold}': AP})
    return result_dict

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    dataset = build_dataset(cfg.eval_config)

    dataset[0]
    scene_name2idx = {}
    scene_name2token = {}
    for idx, sample in enumerate(dataset.samples):
        scene = sample['scene_name']
        token = sample['token']
        if scene not in scene_name2idx:
            scene_name2idx[scene] = []
            scene_name2token[scene] = []
        scene_name2idx[scene].append(idx)
        scene_name2token[scene].append(token)
    all_scene_names = sorted(list(scene_name2idx.keys()))

    gt_matching_path = cfg.eval_config.ann_file.replace('.pkl','_gt_tracks.pkl',)
    with open(gt_matching_path,'rb') as pf:
        gt_matching = pickle.load(pf)
    

    pred_matching_path = args.result_path
    with open(pred_matching_path,'rb') as ppf:
        pred_matching_result_raw = pickle.load(ppf)

    roi_size = torch.tensor(cfg.roi_size).numpy()
    origin = torch.tensor(cfg.pc_range[:2]).numpy()

    if roi_size[0] == 60:
        thresholds_list = [0.5,1.0,1.5]
    elif roi_size[0] == 100:
        thresholds_list = [1.0, 1.5, 2.0]
    else:
        raise ValueError('roi size {} not supported, check again...'.format(roi_size))

    if 'newsplit' in args.result_path:
        gts = get_gts(dataset,new_split=True)
    else:
        gts = get_gts(dataset)

    ### interpolate vector data
    start_time = time.time()
    denormed_gts,pred_matching_result,num_gts,num_preds = \
        get_data(pred_matching_result_raw,gts,origin,roi_size,INTERP_NUM,result_path=args.result_path,denorm=False)
    print('Preparing Data Time {}'.format(time.time()-start_time))

    ### obtain mAP for each threshold
    scene_name_list = []
    for single_scene_name in all_scene_names:
        scene_name_list.append( (single_scene_name,args) )
    result_dict = {thr:{} for thr in thresholds_list}
    for threshold in thresholds_list:
        tpfp_score_list =[]
        for (scene_name,args) in scene_name_list:
            tpfp_score = get_tpfp_from_scene_single(scene_name,args,scene_name2token,pred_matching_result,
                        denormed_gts,gt_matching,threshold)
            tpfp_score_list.append(tpfp_score)
        result_dict[threshold] = get_mAP(tpfp_score_list,num_gts,threshold)
        print(result_dict[threshold])
    
    cat_mean_AP = np.array([0.,0.,0.])
    mean_AP = 0
    for thr in thresholds_list:
        for cat_name in cat2id.keys():
            mean_AP += result_dict[thr][cat_name]['AP@{}'.format(thr)]
            cat_mean_AP[cat2id[cat_name]] += result_dict[thr][cat_name]['AP@{}'.format(thr)]

    cat_map_dict = {cat:cat_mean_AP[idx]/len(thresholds_list) for cat,idx in cat2id.items() }
    print('Category mean AP',cat_map_dict)
    print('mean AP ',mean_AP/(len(cat2id)*len(thresholds_list)))
    print('Overall Time',time.time()-start_time)

if __name__ == '__main__':
    main()