import argparse
import mmcv
from mmcv import Config
import os
from mmdet3d.datasets import build_dataset
import cv2
import torch
import numpy as np
import imageio
import pickle
from functools import partial
from multiprocessing import Pool
import time
from cmap_utils.utils import *
from cmap_utils.match_utils import get_prev2curr_matrix, find_matchings_iou, get_consecutive_vectors,filter_vectors

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

N_WORKERS = 10

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
        '--cons_frames',
        default=5,
        type=int,
        help='consective frames for matchings'
    )
    parser.add_argument(
        '--visual',
        default=0,
        type=int,
        help='whether to visual'
    )
    args = parser.parse_args()
    return args

def match_two_consecutive_frames_pred(args,prev_data,prev_meta,  curr_data, curr_meta,roi_size, origin, cfg):

    prev2curr_matrix = get_prev2curr_matrix(prev_meta,curr_meta)

    prev_vectors = filter_vectors(prev_data,origin,roi_size,args.thr)
    curr_vectors = filter_vectors(curr_data,origin,roi_size,args.thr)

    prev_vectors, curr_vectors, prev2curr_vectors = get_consecutive_vectors(prev_vectors,curr_vectors,
                                    prev2curr_matrix,origin,roi_size) 

    prev2curr_masks, prev2curr_viz = draw_instance_masks(prev2curr_vectors, roi_size, origin, cfg)
    curr_masks, curr_viz = draw_instance_masks(curr_vectors, roi_size, origin, cfg)

    prev2curr_matchings = find_matchings_iou(prev2curr_masks, curr_masks, thresh=0.001)
    curr2prev_matchings = {label:[match_info[1],match_info[0]]  for label,match_info in prev2curr_matchings.items()}
    return curr2prev_matchings

def collect_pred(data,thr):
    vectors = {label: [] for label in cat2id.values()}
    scores = {label: [] for label in cat2id.values()}
    for i in range(len(data['labels'])):
        score, label, v = data['scores'][i], data['labels'][i], data['vectors'][i]
        if score > thr:
            vectors[label].append(np.array(v))
            scores[label].append(score)
    return vectors, scores

def get_scene_matching_result(args,cfg,pred_results,dataset,origin,roi_size,
                              scene_name2idx):
    ### obtain local id sequence matching results of predictions
    vectors_seq = []
    scores_seq = []

    ids_seq = []
    global_map_index = {
        0: 0,
        1: 0,
        2: 0,
    }
    frame_token_list = []
    pred_data_list = []
    meta_list = []

    for idx in scene_name2idx:
        token = dataset[idx]['img_metas'].data['token']
        pred_data = pred_results[token]
        frame_token_list.append(token)
        meta_list.append(dataset[idx]['img_metas'].data)
        pred_data_list.append(pred_data)

    for local_idx in range(len(frame_token_list)):
        curr_pred_data = pred_data_list[local_idx]
        vectors_info, scores = collect_pred(curr_pred_data,args.thr)
        vectors_seq.append(vectors_info)
        scores_seq.append(scores)

        ### assign global id for the first frame
        if local_idx == 0:
            ids_0 = dict()
            for label, vectors in vectors_info.items():
                id_mapping = dict()
                for i, _ in enumerate(vectors):
                    id_mapping[i] = global_map_index[label]
                    global_map_index[label] += 1
                ids_0[label] = id_mapping
            ids_seq.append(ids_0)
            continue

        ### from the farthest to the nearest
        history_range = range(max(local_idx-args.cons_frames,0),local_idx)
        tmp_ids_list = []
        for comeback_idx,prev_idx in enumerate(history_range):

            tmp_ids = {label:{} for label in cat2id.values()} 
            curr_pred_data = pred_data_list[local_idx]
            comeback_pred_data = pred_data_list[prev_idx]
            curr_meta = meta_list[local_idx]
            comeback_meta = meta_list[prev_idx]

            curr2prev_matching = match_two_consecutive_frames_pred(args,comeback_pred_data,comeback_meta,
                                            curr_pred_data, curr_meta,roi_size, origin, cfg)
            
            for label,match_info in curr2prev_matching.items():
                for curr_match_local_idx,prev_match_local_idx in enumerate(match_info[0]):
                    if prev_match_local_idx == -1:
                        tmp_ids[label][curr_match_local_idx] = -1
                    else:
                        prev_match_global_idx = ids_seq[prev_idx][label][prev_match_local_idx]
                        tmp_ids[label][curr_match_local_idx] = prev_match_global_idx

            tmp_ids_list.append(tmp_ids)

        ids_n = {label:{} for label in cat2id.values()}

        ### assign global id based on previous k frames' global id
        missing_matchings = {label:[] for label in cat2id.values()}
        for tmp_match in tmp_ids_list[::-1]:
            for label, matching in tmp_match.items():
                for vec_local_idx, vec_glb_idx in matching.items():
                    if vec_local_idx not in ids_n[label].keys():
                        if vec_glb_idx != -1 and vec_glb_idx not in ids_n[label].values():
                            ids_n[label][vec_local_idx] = vec_glb_idx
                            if vec_local_idx in missing_matchings[label]:
                                missing_matchings[label].remove(vec_local_idx)
                        else:
                            missing_matchings[label].append(vec_local_idx)

        ### assign new id if one vector is not matched 
        for label,miss_match in missing_matchings.items():
            for miss_idx in miss_match:
                if miss_idx not in ids_n[label].keys():
                    ids_n[label][miss_idx] = global_map_index[label]
                    global_map_index[label] += 1
        ids_seq.append(ids_n)

    return ids_seq, vectors_seq, scores_seq, meta_list

def generate_results(ids_info,vectors_seq,scores_seq,meta_list,scene_name):
    ### assign global id 

    global_gt_idx = {}
    result_list = []
    instance_count = 0
    for f_idx in range(len(ids_info)):
        output_dict = {'vectors':[],'global_ids':[],'labels':[],'scores':[],'local_idx':[]}
        output_dict['scene_name'] = scene_name
        output_dict['meta'] = meta_list[f_idx]
        for label in cat2id.values():
            for local_idx, global_label_idx in ids_info[f_idx][label].items():
                overall_count_idx = label*100 + global_label_idx
                if overall_count_idx not in global_gt_idx.keys():
                    overall_global_idx = instance_count
                    global_gt_idx[overall_count_idx] = overall_global_idx
                    instance_count += 1
                else:
                    overall_global_idx = global_gt_idx[overall_count_idx]
                output_dict['global_ids'].append(overall_global_idx)
                output_dict['vectors'].append(vectors_seq[f_idx][label][local_idx])
                output_dict['scores'].append(scores_seq[f_idx][label][local_idx])
                output_dict['labels'].append(label)
        output_dict['local_idx'] = f_idx

        result_list.append(output_dict)
    return result_list

def get_matching_single(scene_name,args,scene_name2idx,dataset,cfg,pred_results,origin,roi_size):
    name2idx = scene_name2idx[scene_name]
    ids_info, vectors_seq,scores_seq,meta_list = get_scene_matching_result(args,cfg,pred_results,dataset,
            origin,roi_size,name2idx)
    gen_result = generate_results(ids_info,vectors_seq,scores_seq,meta_list,scene_name)

    return (scene_name,ids_info,gen_result)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    dataset = build_dataset(cfg.match_config)

    scene_name2idx = {}
    scene_name2token = {}
    for idx, sample in enumerate(dataset.samples):
        scene = sample['scene_name']
        token = sample['token']
        if scene not in scene_name2idx:
            scene_name2idx[scene] = []
            scene_name2token[scene] = []
        scene_name2idx[scene].append(idx)

    submission = mmcv.load(args.result_path)
    results = submission['results']

    all_scene_names = sorted(list(scene_name2idx.keys()))
    all_scene_matching_meta = {}

    scene_info_list = []

    for single_scene_name in all_scene_names:
        scene_info_list.append( (single_scene_name,args) )

    roi_size = torch.tensor(cfg.roi_size).numpy()
    origin = torch.tensor(cfg.pc_range[:2]).numpy()

    start_time = time.time()

    if N_WORKERS > 0:
        fn = partial(get_matching_single, scene_name2idx=scene_name2idx,dataset=dataset,cfg=cfg,
                    pred_results=results,origin=origin,roi_size=roi_size)
        pool = Pool(N_WORKERS)
        matching_results = pool.starmap(fn,scene_info_list)
        pool.close()
    else:
        matching_results =[]
        for scene_info in scene_info_list:
            scene_name = scene_info[0]
            single_matching_result = get_matching_single(scene_name=scene_name, scene_name2idx=scene_name2idx,
                    args=args,  dataset=dataset,cfg=cfg,pred_results=results,origin=origin,roi_size=roi_size)
            matching_results.append(single_matching_result)

    final_reuslt = []
    for single_matching_info in matching_results:
        scene_name = single_matching_info[0]
        single_matching = single_matching_info[1]
        all_scene_matching_meta[scene_name] = single_matching
        final_reuslt.extend(single_matching_info[2])

    meta_path = args.result_path.replace('submission_vector.json','pos_predictions_{}.pkl'.format(args.cons_frames))
    with open(meta_path, 'wb') as f:
        pickle.dump(final_reuslt, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Matching Time',time.time()-start_time)


if __name__ == '__main__':
    main()