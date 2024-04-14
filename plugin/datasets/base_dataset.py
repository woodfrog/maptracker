import numpy as np
import os
import os.path as osp
import mmcv
from .evaluation.raster_eval import RasterEvaluate
from .evaluation.vector_eval import VectorEvaluate
from mmdet3d.datasets.pipelines import Compose
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from mmcv.parallel import DataContainer as DC
import warnings
import pickle


warnings.filterwarnings("ignore")

@DATASETS.register_module()
class BaseMapDataset(Dataset):
    """Map dataset base class.

    Args:
        ann_file (str): annotation file path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        eval_config (Config): evaluation config
        meta (dict): meta information
        pipeline (Config): data processing pipeline config,
        interval (int): annotation load interval
        work_dir (str): path to work dir
        test_mode (bool): whether in test mode
    """
    def __init__(self, 
                 ann_file,
                 cat2id,
                 roi_size,
                 meta,
                 pipeline,
                 interval=1,
                 seq_split_num=1,
                 work_dir=None,
                 eval_config=None,
                 test_mode=False,
                 multi_frame=False,
                 sampling_span=10,
                 matching=False,
                 eval_semantic=False,
        ):
        super().__init__()
        self.ann_file = ann_file
        self.multi_frame = multi_frame
        self.sampling_span = sampling_span
        self.matching = matching
        
        self.meta = meta
        self.classes = list(cat2id.keys())
        self.num_classes = len(self.classes)
        self.cat2id = cat2id
        self.interval = interval
        self.seq_split_num = seq_split_num
        self.eval_semantic = eval_semantic

        self.load_annotations(self.ann_file)

        if matching:
            assert self.multi_frame, 'The matching info has to loaded under the multi-frame setting'
            self.matching_file = ann_file[:-4] + '_gt_tracks.pkl'
            assert os.path.isfile(self.matching_file)
            self.load_matching(self.matching_file)
        
        self.idx2token = {}
        for i, s in enumerate(self.samples):
            self.idx2token[i] = s['token']
        self.token2idx = {v: k for k, v in self.idx2token.items()}

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None
        
        # dummy flags to fit with mmdet dataset
        self.flag = np.zeros(len(self), dtype=np.uint8)

        self.roi_size = roi_size
        
        self.work_dir = work_dir
        self.eval_config = eval_config
        if self.eval_config is not None:
            assert test_mode, "eval_config is valid only in test_mode"

        # record the sequence information, prepare for two-frame data loading
        self._set_sequence_info()
        
        self._set_sequence_group_flag()
        
    
    def _set_sequence_info(self):
        """Compute and record the sequence id and local index of each sample
        """
        scene_name2idx = {}
        for idx, sample in enumerate(self.samples):
            self.samples[idx]['modified_sample_idx'] = idx
            scene = sample['scene_name']
            if scene not in scene_name2idx:
                scene_name2idx[scene] = []
                self.samples[idx]['prev'] = -1

            scene_name2idx[scene].append(idx)
        self.scene_name2idx = scene_name2idx

        print('Prepare sequence information for {}'.format(self.ann_file))
        idx2scene = {}
        for scene_name, scene_info in scene_name2idx.items():
            for local_idx, global_idx in enumerate(scene_info):
                idx2scene[global_idx] = (scene_name, local_idx, len(scene_info))
        self.idx2scene = idx2scene

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        if self.seq_split_num == -1:
            self.flag = np.arange(len(self.samples))
            return
        elif self.seq_split_num == -2:
            return
        
        res = []

        curr_sequence = -1
        for idx in range(len(self.samples)):
            if self.samples[idx]['prev'] == -1:
                # new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            bin_counts = np.bincount(self.flag)
            new_flags = []
            curr_new_flag = 0
            for curr_flag in range(len(bin_counts)):
                seq_length = int(round(bin_counts[curr_flag] / self.seq_split_num))
                curr_sequence_length = list(range(0, bin_counts[curr_flag], seq_length)) + [bin_counts[curr_flag]]
                
                # if left one sample, put it into the last sequence
                if curr_sequence_length[-1] - curr_sequence_length[-2] <= 1:
                    curr_sequence_length = curr_sequence_length[:-2] + [curr_sequence_length[-1]]
                
                curr_sequence_length = np.array(curr_sequence_length)

                for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                    for _ in range(sub_seq_idx):
                        new_flags.append(curr_new_flag)
                    curr_new_flag += 1

            assert len(new_flags) == len(self.flag)
            # assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
            self.flag = np.array(new_flags, dtype=np.int64)

    def load_annotations(self, ann_file):
        raise NotImplementedError
    
    def load_matching(self, matching_file):
        raise NotImplementedError

    def get_sample(self, idx):
        raise NotImplementedError

    def format_results(self, results, denormalize=True, prefix=None, save_semantic=False):
        '''Format prediction result to submission format.
        
        Args:
            results (list[Tensor]): List of prediction results.
            denormalize (bool): whether to denormalize prediction from (0, 1) \
                to bev range. Default: True
            prefix (str): work dir prefix to save submission file.

        Returns:
            dict: Evaluation results
        '''

        meta = self.meta
        output_format = meta['output_format']
        submissions = {
            'meta': meta,
            'results': {},
        }

        if output_format == 'raster':
            for pred in results:
                single_case = {}
                token = pred['token']
                pred_map = pred['semantic_mask']
                pred_bool = pred_map > 0
                single_case['semantic_mask'] = pred_bool.bool()
                submissions['results'][token] = single_case
            
            # Use pickle format to minimize submission file size.
            out_path = osp.join(prefix, 'submission_raster.pkl')
            print(f'saving submissions results to {out_path}')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            mmcv.dump(submissions, out_path)
            return out_path

        elif output_format == 'vector':
            all_pos_results = []
            for pred in results:
                '''
                For each case, the result should be formatted as Dict{'vectors': [], 'scores': [], 'labels': []}
                'vectors': List of vector, each vector is a array([[x1, y1], [x2, y2] ...]),
                    contain all vectors predicted in this sample.
                'scores: List of score(float), 
                    contain scores of all instances in this sample.
                'labels': List of label(int), 
                    contain labels of all instances in this sample.
                '''
                if pred is None: # empty prediction
                    continue
                
                single_case = {'vectors': [], 'scores': [], 'labels': [], 'props': [],
                        'track_vectors': [], 'track_scores': [], 'track_labels': []}
                token = pred['token']
                roi_size = np.array(self.roi_size)
                origin = -np.array([self.roi_size[0]/2, self.roi_size[1]/2])
                
                # save the extra semantic info
                if save_semantic:
                    single_case['semantic_mask'] = pred['semantic_mask'].tolist()

                if 'scores' in pred:
                    for i in range(len(pred['scores'])):
                        score = pred['scores'][i]
                        label = pred['labels'][i]
                        vector = pred['vectors'][i]
                        prop = pred['props'][i]

                        # A line should have >=2 points
                        if len(vector) < 2:
                            continue
                        
                        if denormalize:
                            eps = 1e-5
                            vector = vector * (roi_size + eps) + origin

                        single_case['vectors'].append(vector)
                        single_case['scores'].append(score)
                        single_case['labels'].append(label)
                        single_case['props'].append(prop)
                
                if 'track_scores' in pred:
                # also save the tracking information for analyzing
                    for i in range(len(pred['track_scores'])):
                        score = pred['track_scores'][i]
                        label = pred['track_labels'][i]
                        vector = pred['track_vectors'][i]
                        if denormalize:
                            eps = 1e-5
                            vector = vector * (roi_size + eps) + origin
                        single_case['track_vectors'].append(vector)
                        single_case['track_scores'].append(score)
                        single_case['track_labels'].append(label)

                submissions['results'][token] = single_case
                
                if not self.eval_semantic:
                    pos_results = pred['pos_results']
                    pos_vectors = pos_results['vectors']
                    if denormalize and len(pos_vectors) > 0:
                        pos_vectors = pos_vectors.reshape(pos_vectors.shape[0], -1, 2)
                        pos_vectors = (pos_vectors * roi_size + origin).reshape(pos_vectors.shape[0], -1)
                    save_pos_results = {
                        'vectors': pos_vectors,
                        'labels': pos_results['labels'],
                        'scores': pos_results['scores'],
                        'scene_name': pos_results['scene_name'],
                        'local_idx': pos_results['local_idx'],
                        'global_ids': pos_results['global_ids'],
                        'meta': pred['meta']
                    }
                    all_pos_results.append(save_pos_results)
                
            out_path = osp.join(prefix, 'submission_vector.json')
            print(f'saving submissions results to {out_path}')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            mmcv.dump(submissions, out_path)

            if not self.eval_semantic:
                out_path_pos = osp.join(prefix, 'pos_predictions.pkl')
                with open(out_path_pos, 'wb') as f:
                    pickle.dump(all_pos_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            return out_path

        else:
            raise ValueError("output format must be either \'raster\' or \'vector\'")

    def evaluate(self, results, logger=None, **kwargs):
        '''Evaluate prediction result based on `output_format` specified by dataset.

        Args:
            results (list[Tensor]): List of prediction results.
            logger (logger): logger to print evaluation results.

        Returns:
            dict: Evaluation results.
        '''
        print('len of the results', len(results))

        eval_semantic = True if (hasattr(self, 'eval_semantic') and self.eval_semantic) else False
        save_semantic = True if 'save_semantic' in kwargs and kwargs['save_semantic'] or eval_semantic \
                            else False
        
        result_path = self.format_results(results, denormalize=True, 
                        prefix=self.work_dir, save_semantic=save_semantic)

        return self._evaluate(result_path, logger=logger, eval_semantic=eval_semantic)
    
    def _evaluate(self, result_path, logger=None, eval_semantic=False):
        if not eval_semantic:
            self.evaluator = VectorEvaluate(self.eval_config)
        else:
            self.evaluator = RasterEvaluate(self.eval_config)
        result_dict = self.evaluator.evaluate(result_path, logger=logger)
        return result_dict

    def show_gt(self, idx, out_dir='demo/'):
        '''Visualize ground-truth.

        Args:
            idx (int): index of sample.
            out_dir (str): output directory.
        '''

        from mmcv.parallel import DataContainer
        from copy import deepcopy
        sample = self.get_sample(idx)
        sample = deepcopy(sample)
        data = self.pipeline(sample)

        #imgs = [mmcv.imread(i) for i in sample['img_filenames']]
        #cam_extrinsics = sample['cam_extrinsics']
        #cam_intrinsics = sample['cam_intrinsics']

        if 'vectors' in data:
            vectors = data['vectors']
            if isinstance(vectors, DataContainer):
                vectors = vectors.data

            self.renderer.render_bev_from_vectors(vectors, out_dir)
            #self.renderer.render_camera_views_from_vectors(vectors, imgs, 
            #    cam_extrinsics, cam_intrinsics, 2, out_dir)

        if 'semantic_mask' in data:
            semantic_mask = data['semantic_mask']
            if isinstance(semantic_mask, DataContainer):
                semantic_mask = semantic_mask.data
            
            self.renderer.render_bev_from_mask(semantic_mask, out_dir, flip=True)

    def show_result(self, submission, idx, score_thr=0, draw_score=False, show_semantic=False, out_dir='demo/'):
        '''Visualize prediction result.

        Args:
            idx (int): index of sample.
            submission (dict): prediction results.
            score_thr (float): threshold to filter prediction results.
            out_dir (str): output directory.
        '''

        meta = submission['meta']
        output_format = meta['output_format']
        token = self.idx2token[idx]
        results = submission['results'][token]
        sample = self.get_sample(idx)


        if 'semantic_mask' in results and show_semantic:
            semantic_mask = np.array(results['semantic_mask'])
            self.renderer.render_bev_from_mask(semantic_mask, out_dir, flip=False)
        
        if output_format == 'vector' and 'scores' in results:
            vectors = {label: [] for label in self.cat2id.values()}
            for i in range(len(results['labels'])):
                score = results['scores'][i]
                label = results['labels'][i]
                prop = results['props'][i]
                v = results['vectors'][i]

                if score > score_thr:
                    if draw_score:
                        vectors[label].append((v, score, prop))
                    else:
                        vectors[label].append(v)

            self.renderer.render_bev_from_vectors(vectors, out_dir, draw_scores=draw_score)

            # For projecting and visualizing results on perspective images
            #imgs = [mmcv.imread(i) for i in sample['img_filenames']]
            #cam_extrinsics = sample['cam_extrinsics']
            #cam_intrinsics = sample['cam_intrinsics']
            # self.renderer.render_camera_views_from_vectors(vectors, imgs, 
            #         cam_extrinsics, cam_intrinsics, 2, out_dir)
    

    def show_track(self, submission, idx, out_dir='demo/'):
        '''Visualize prediction result.

        Args:
            idx (int): index of sample.
            submission (dict): prediction results.
            score_thr (float): threshold to filter prediction results.
            out_dir (str): output directory.
        '''

        meta = submission['meta']
        token = self.idx2token[idx]
        results = submission['results'][token]

        vectors = {label: [] for label in self.cat2id.values()}
        for i in range(len(results['track_labels'])):
            score = results['track_scores'][i]
            label = results['track_labels'][i]
            v = results['track_vectors'][i]
            vectors[label].append((v, score, 1))
        
        self.renderer.render_bev_from_vectors(vectors, out_dir, draw_scores=True)

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.samples)
        
    def _rand_another(self, idx):
        """Randomly get another item.

        Returns:
            int: Another index of item.
        """
        return np.random.choice(self.__len__)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        input_dict = self.get_sample(idx)
        data = self.pipeline(input_dict)

        # prepare the local sequence index info
        seq_info = self.idx2scene[idx]
        data['seq_info'] = DC(seq_info, cpu_only=True)

        if self.multi_frame: # used when sampling multi-frame training data
            scene_name = input_dict['scene_name']
            scene_seq_info = self.scene_name2idx[scene_name]
            local_idx_curr = input_dict['sample_idx'] - scene_seq_info[0]

            span = max(self.sampling_span, self.multi_frame)
            min_idx = local_idx_curr - span
            sampled_indices = np.random.choice(span, self.multi_frame-1, replace=False).tolist()
            sampled_indices = sorted(sampled_indices)
            local_indices_prev = [min_idx + x for x in sampled_indices]
            local_indices_prev = [x if x>=0 else 0 for x in local_indices_prev]
            
            data['img_metas'].data['local_idx'] = local_idx_curr
            global_indices_prev = [local_idx + scene_seq_info[0] for local_idx in local_indices_prev]

            all_prev_data = []
            for idx, global_idx_prev in enumerate(global_indices_prev):
                input_dict_prev = self.get_sample(global_idx_prev)
                data_prev = self.pipeline(input_dict_prev)
                local_idx_prev = local_indices_prev[idx]
                data_prev['img_metas'].data['local_idx'] = local_idx_prev
                all_prev_data.append(data_prev)

            all_local2global_info = []
            if self.matching:
                scene_matching_info = self.matching_meta[scene_name]
                for local_idx_prev in local_indices_prev:
                    prev_local2global = DC(scene_matching_info['instance_ids'][local_idx_prev], cpu_only=True)
                    all_local2global_info.append(prev_local2global)
            curr_local2global = DC(scene_matching_info['instance_ids'][local_idx_curr], cpu_only=True)
            all_local2global_info.append(curr_local2global)
                    
            data['all_prev_data'] = all_prev_data
            data['all_local2global_info'] = all_local2global_info
        
        return data

