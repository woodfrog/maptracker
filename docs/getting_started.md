# Getting started with MapTracker

In this document, we provide the commands for running inference/evaluation, training, and visualization.


## Inference and evaluation


### Inference and evaluate with Chamfer-based mAP


Run the following command to do inference and evaluation using the pretrained checkpoints, assuming 8 GPUs are used.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  bash tools/dist_test.sh  plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py    work_dirs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth  8  --eval --eval-options save_semantic=True
```

Set the ``--eval-options save_semantic=True`` to also save the semantic segmentation results of the BEV module.


### Evaluate with C-mAP

Generate prediction matching by
```
python tools/tracking/prepare_pred_tracks.py ${CONFIG} --result_path ${SUBMISSION_FILE} --cons_frames ${COMEBACK_FRAMES}
```

Evaluate with C-mAP by
```
python tools/tracking/calculate_cmap.py ${CONFIG} --result_path ${PRED_MATCHING_INFO}
```

An example evaluation:
```
python tools/tracking/calculate_cmap.py plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py --result_path ./work_dirs/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions.pkl
```

### Results

By running with the checkpoints we provided in the [data preparation guide](docs/data_preparation.md), the expected results are:

|                          Dataset                               | Split | Divider | Crossing | Boundary | mAP |      C-mAP  |
|:------------------------------------------------------------------------:|:--------:|:-------:|:--------:|:--------:|:---------:|:-------------------------------------------------------------------------------------------:|
|            nuScenes             |  old  |  74.14  |  80.04   |  74.06   |   76.08  | 69.13  |
|            nuScenes             |  new  |  30.10  |  45.86   |  45.06   |   40.34  | 32.50  |
|            Argoverse2           |  old  |  76.99  |  79.97   |  73.66   |   76.87  | 68.35  |
|            Argoverse2           |  new  |  75.11  |  69.96   |  68.95   |   71.34  | 63.11  |


## Training

The training consists of three stages as detailed in the paper. We train the models on 8 Nvidia RTX A5000 GPUs. 

**Stage 1**: BEV pretraining with semantic segmentation losses:
```
bash ./tools/dist_train.sh plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage1_bev_pretrain.py 8
```

**Stage 2**: Vector module warmup with a large batch size while freezing the BEV module:
```
bash ./tools/dist_train.sh plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup.py 8
```
Set up the ``load_from=...`` properly in the config file to load the checkpoint from stage 1.

**Stage 3**: Joint finetuning:
```
bash ./tools/dist_train.sh plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py 8
```
Set up the ``load_from=...`` properly in the config file to load the checkpoint from stage 2.



## Visualization

### Global merged reconstruction (merged from local HD maps)

```bash
python tools/visualization/teaser_vis.py [path to method configuration file under plugin/configs] \
  --data_path [path to the .pkl file] \
  --out_dir [path to the output folder] \
  --option [vis-pred / vis-gt: visualize predicted vectors / visualize ground truth vectors] \
  --per_frame_result 1
```
Set the ``--per_frame_result`` to 1 to generate the per-frame video, the visualization is a bit slow; set it to 0 to only produce the final merged global reconstruction. 


Examples:
```bash
# Visualize MapTracker's prediction
python tools/visualization/teaser_vis.py plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py \
--data_path work_dirs/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions.pkl \
--out_dir vis_global/nuscenes_old/maptracker \
--option vis-pred  --per_frame_result 1

# Visualize groud truth data
python tools/visualization/teaser_vis.py plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py \
--data_path datasets/nuscenes/nuscenes_map_infos_val_gt_tracks.pkl \
--out_dir vis_global/nuscenes_old/gt  \
--option vis-gt --per_frame_result 0
```


### Local HD map reconstruction

```bash
python tools/visualization/teaser_vis_per_frame.py [path to method configuration file under plugin/configs] \
  --data_path [path to the .pkl file] \
  --out_dir [path to the data folder] \
  --option [vis-pred / vis-gt: visualize predicted vectors / visualize ground truth vectors and input video streams]
```

Note that the input perspective-view videos will be saved when generating the ground truth visualization.


Examples:
```bash
# Visualize MapTracker's prediction
python tools/visualization/teaser_vis_per_frame.py plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py \
--data_path work_dirs/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions.pkl \
--out_dir vis_local/nuscenes_old/maptracker \
--option vis-pred

# Visualize groud truth data
python tools/visualization/teaser_vis_per_frame.py plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py \
--data_path datasets/nuscenes/nuscenes_map_infos_val_gt_tracks.pkl \
--out_dir vis_local/nuscenes_old/gt  \
--option vis-gt
```
