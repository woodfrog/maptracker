"""Debug script to check if local_idx is being set correctly and reaching >= 2"""
import sys
import torch
from mmdet.datasets import build_dataloader, build_dataset
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Build dataset with training config
cfg_path = 'plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py'

from mmcv import Config
cfg = Config.fromfile(cfg_path)

# Build dataset for training
dataset = build_dataset(cfg.data.train)

logger.info(f"Total samples in dataset: {len(dataset)}")
logger.info(f"Multi-frame setting: {dataset.multi_frame}")
logger.info(f"Sampling span: {dataset.sampling_span}" if dataset.multi_frame else "Not using multi-frame")

# Sample 100 batches and check local_idx values
local_idx_values = []
local_idx_curr_values = []

for i in range(min(100, len(dataset))):
    try:
        data = dataset[i]
        if 'img_metas' in data and hasattr(data['img_metas'], 'data'):
            local_idx = data['img_metas'].data.get('local_idx', -1)
            local_idx_values.append(local_idx)
            local_idx_curr_values.append(local_idx)
            
            if i < 20:  # Print first 20
                logger.info(f"Sample {i}: local_idx = {local_idx}")
    except Exception as e:
        logger.error(f"Error at sample {i}: {e}")
        continue

if local_idx_values:
    logger.info(f"\n=== LOCAL_IDX STATISTICS ===")
    logger.info(f"Min local_idx: {min(local_idx_values)}")
    logger.info(f"Max local_idx: {max(local_idx_values)}")
    logger.info(f"Mean local_idx: {sum(local_idx_values)/len(local_idx_values):.2f}")
    logger.info(f"Values >= 2: {sum(1 for x in local_idx_values if x >= 2)} / {len(local_idx_values)}")
    logger.info(f"Unique values: {sorted(set(local_idx_values))}")
    
    # Check if corruption should be active
    if any(x >= 2 for x in local_idx_values):
        logger.warning("✓ Some samples have local_idx >= 2, so corruption SHOULD be activating")
    else:
        logger.error("✗ NO samples have local_idx >= 2, corruption will NEVER activate!")
else:
    logger.error("No local_idx values found!")
