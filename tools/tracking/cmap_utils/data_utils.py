import mmcv
import os
from mmdet3d.datasets import build_dataloader
import numpy as np
from copy import deepcopy
from functools import partial
from multiprocessing import Pool

from .utils import *
from .match_utils import *

cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}

def get_gts(dataset,new_split=False,N_WORKERS=16):
    roi_size = dataset.roi_size
    if 'av2' in dataset.ann_file:
        dataset_name = 'av2'
    else:
        dataset_name = 'nusc'
    if new_split:
        tmp_file = f'./tmp_gts_{dataset_name}_{roi_size[0]}x{roi_size[1]}_newsplit.pkl'
    else:
        tmp_file = f'./tmp_gts_{dataset_name}_{roi_size[0]}x{roi_size[1]}.pkl'
    if os.path.exists(tmp_file):
        print(f'loading cached gts from {tmp_file}')
        gts = mmcv.load(tmp_file)
    else:
        print('collecting gts...')
        gts = {}
        # pdb.set_trace()
        dataloader = build_dataloader(
            dataset, samples_per_gpu=1, workers_per_gpu=N_WORKERS, shuffle=False, dist=False)
        pbar = mmcv.ProgressBar(len(dataloader))
        for data in dataloader:
            token = deepcopy(data['img_metas'].data[0][0]['token'])
            gt = deepcopy(data['vectors'].data[0][0])
            # pdb.set_trace()
            gts[token] = gt
            pbar.update()
            del data # avoid dataloader memory crash
    
    for token, gt in gts.items():
        for label, vectors in gt.items():
            label_vecs = []
            for vec in vectors:
                label_vecs.append(interp_fixed_num(vec,20))
            gt[label] = label_vecs
        gts[token] = gt
    return gts

def prepare_data_multi(token,idx,pred,gts,origin,roi_size,interp_num,dataset,denorm=False):
    num_gts = np.array([0,0,0])
    num_preds = np.array([0,0,0])
    denorm_gt = {}

    gt = gts[token]
    denorm_gt = {label:[] for label in cat2id.values()}
    scores_by_cls = {label: [] for label in cat2id.values()}

    vector_list = []
    for i in range(len(pred['labels'])):
        score = pred['scores'][i]
        vector = pred['vectors'][i].reshape(-1,2)
        label = pred['labels'][i]
        scores_by_cls[label].append(score)
        if not denorm:
            vector_list.append(interp_fixed_num(vector,interp_num))
        else:
            vector_list.append(interp_fixed_num(vector*roi_size+origin,interp_num))

    for label in cat2id.values():
        for vec in gt[label]:
            denorm_gt[label].append(interp_fixed_num(vec,interp_num))

    for label in cat2id.values():
        num_gts[label] += len(gt[label])
        num_preds[label] += len(scores_by_cls[label])
    return token,idx,denorm_gt, vector_list, num_gts,num_preds

def get_data(pred_matching_result_raw,gts,origin,roi_size,num_interp,result_path,denorm=False):
    ### collect data, interpolate with multi_processing
    token_list = []
    for idx,pred_res in enumerate(pred_matching_result_raw):
        token = pred_res['meta']['token']
        token_list.append( (token,idx,pred_matching_result_raw[idx]) )
    dataset = 'av2' if 'av2' in result_path else 'nusc'
    fn = partial(prepare_data_multi,gts=gts,origin=origin,roi_size=roi_size,interp_num=num_interp,dataset=dataset,denorm=denorm)

    denormed_gts = {}
    pred_matching_result = {}
    num_gts = np.zeros(3)
    num_preds = np.zeros(3)
    with Pool(processes=16) as pool:
        data_infos = pool.starmap(fn,token_list)
    for data_info in data_infos:
        token,idx, denorm_gt,pred_vector, num_gts_single,num_preds_single = data_info
        denormed_gts[token] = denorm_gt
        pred_matching_result_raw[idx]['vectors'] = pred_vector
        pred_matching_result[token] = pred_matching_result_raw[idx]
        num_gts  = num_gts + num_gts_single
        num_preds = num_preds + num_preds_single


    return denormed_gts,pred_matching_result,num_gts,num_preds