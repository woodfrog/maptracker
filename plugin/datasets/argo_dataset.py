from .base_dataset import BaseMapDataset
from .map_utils.av2map_extractor import AV2MapExtractor
from mmdet.datasets import DATASETS
import numpy as np
from .visualize.renderer import Renderer
from time import time
import mmcv
from pyquaternion import Quaternion

import pickle
import os


@DATASETS.register_module()
class AV2Dataset(BaseMapDataset):
    """Argoverse2 map dataset class.

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

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        self.map_extractor = AV2MapExtractor(self.roi_size, self.id2map)

        self.renderer = Renderer(self.cat2id, self.roi_size, 'av2')
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        
        start_time = time()
        ann = mmcv.load(ann_file)
        self.id2map = ann['id2map']
        samples = ann['samples']

        if 'newsplit' not in ann_file:
            if 'val' in ann_file:
                # For the old split, we make sure that the test set matches exactly with the MapTR codebase
                # NOTE: simply sort&sampling will produce slightly different results compared to MapTR's samples
                # so we have to directly use the saved meta information from MapTR codebase to get the samples
                maptr_meta_path = os.path.join(os.path.dirname(ann_file), 'maptrv2_val_samples_info.pkl')
                with open(maptr_meta_path, 'rb') as f:
                    maptr_meta = pickle.load(f)
                maptr_unique_tokens = [x['token'] for x in maptr_meta['samples_meta']]

                unique_token2samples = {}
                for sample in samples:
                    unique_token2samples[f'{sample["log_id"]}_{sample["token"]}'] = sample

                samples = [unique_token2samples[x] for x in maptr_unique_tokens]
            else:
                # For the old split, we follow MapTR's data loading, which
                # sorts the samples based on the token
                samples = list(sorted(samples, key=lambda e: e['token']))
                samples = samples[::self.interval]
        else:
            # For the new split, we simply follow StreamMapNet, do not sort based on the token
            # In this way, the intervals between consecutive frames are uniform...
            samples = samples[::self.interval]

        # Since the sorted order copied from MapTR does not strictly enforce that
        # samples of the same scene are consecutive, need to re-arrange
        scene_name2idx = {}
        for idx, sample in enumerate(samples):
            scene = sample['log_id']
            if scene not in scene_name2idx:
                scene_name2idx[scene] = []
            scene_name2idx[scene].append(idx)

        samples_rearrange = []
        for scene_name in scene_name2idx:
            scene_sample_ids = scene_name2idx[scene_name]
            for sample_id in scene_sample_ids:
                samples_rearrange.append(samples[sample_id])
        
        samples = samples_rearrange

        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
        self.samples = samples

    def load_matching(self, matching_file):
        with open(matching_file, 'rb') as pf:
            data = pickle.load(pf)
        total_samples = 0
        for scene_name, info in data.items():
            total_samples += len(info['sample_ids'])

        assert total_samples == len(self.samples), 'Matching info not matched with data samples'
        self.matching_meta = data
        print(f'loaded matching meta for {len(data)} scenes')

    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        log_id = sample['log_id']
        map_geoms = self.map_extractor.get_map_geom(log_id, sample['e2g_translation'], 
                sample['e2g_rotation'])

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v
        
        ego2img_rts = []
        for c in sample['cams'].values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)


        # pdb.set_trace()

        input_dict = {
            'token': sample['token'],
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            # extrinsics are 4x4 tranform matrix, NOTE: **ego2cam**
            'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            'ego2global_translation': sample['e2g_translation'], 
            'ego2global_rotation': sample['e2g_rotation'].tolist(),
            'sample_idx': sample['modified_sample_idx'],
            'scene_name': sample['scene_name'],
            'lidar_path': sample['lidar_fpath']
        }

        return input_dict