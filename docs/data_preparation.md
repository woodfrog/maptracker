
# Data Preparation

Compared to the data preparation procedure of StreamMapNet or MapTR, we have one more step to generate the ground truth tracking information (Step 3). 

We noticed that the track generation results can be slighly different when running on different machines (potentially because Shapely's behaviors are slightly different across different machines), **so please always run the Step 3 below on the training machine to generate the gt tracking information**. 

## nuScenes
**Step 1.** Download [nuScenes](https://www.nuscenes.org/download) dataset to `./datasets/nuscenes`.


**Step 2.** Generate annotation files for NuScenes dataset (the same as StreamMapNet)

```
python tools/data_converter/nuscenes_converter.py --data-root ./datasets/nuscenes
```

Add ``--newsplit`` to generate the metadata for the new split (geographical-based split) provided by StreamMapNet.

**Step 3.** Generate the tracking ground truth by 

```
python tools/tracking/prepare_gt_tracks.py plugin/configs/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py  --out-dir tracking_gts/nuscenes --visualize
```

Add the ``--visualize`` flag to visualize the data with element IDs derived from our track generation process, or remove it to save disk memory.  

For generating the G.T. tracks of the new split, change the config file accordingly.


## Argoverse2

**Step 1.** Download [Argoverse2 (sensor)](https://argoverse.github.io/user-guide/getting_started.html#download-the-datasets) dataset to `./datasets/av2`.

**Step 2.** Generate annotation files for Argoverse2 dataset.

```
python tools/data_converter/argoverse_converter.py --data-root ./datasets/av2
```

**Step 3.** Generate the tracking ground truth by 

```
python tools/tracking/prepare_gt_tracks.py plugin/configs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py  --out-dir tracking_gts/av2 --visualize
```


## Checkpoints

We provide the checkpoints at [this link](https://www.dropbox.com/scl/fo/miulg8q9oby7q2x5vemme/ALoxX1HyxGlfR9y3xlqfzeE?rlkey=i3rw4mbq7lacblc7xsnjkik1u&dl=0). Please download and place them as ``./work_dirs/pretrained_ckpts``.


## File structures

Make sure the final file structures look like below:

```
maptracker
├── mmdetection3d
├── tools
├── plugin
│   ├── configs
│   ├── models
│   ├── datasets
│   ├── ...
├── work_dirs
│   ├── pretrained_ckpts
│   │   ├── maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune
│   │   │   ├── latest.pth
│   │   ├── ...
│   ├── ....
├── datasets
│   ├── nuscenes
│   │   ├── maps <-- used
│   │   ├── samples <-- key frames
│   │   ├── v1.0-test <-- metadata
|   |   ├── v1.0-trainval <-- metadata and annotations
│   │   ├── nuscenes_map_infos_train_{newsplit}.pkl <-- train annotations
│   │   ├── nuscenes_map_infos_train_{newsplit}_gt_tracks.pkl <-- train gt tracks
│   │   ├── nuscenes_map_infos_val_{newsplit}.pkl <-- val annotations
│   │   ├── nuscenes_map_infos_val_{newsplit}_gt_trakcs.pkl <-- val gt tracks
│   ├── av2
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── maptrv2_val_samples_info.pkl <-- maptr's av2 metadata, used to align the val set
│   │   ├── av2_map_infos_train_{newsplit}.pkl <-- train annotations
│   │   ├── av2_map_infos_train_{newsplit}_gt_tracks.pkl <-- train gt tracks
│   │   ├── av2_map_infos_val_{newsplit}.pkl <-- val annotations
│   │   ├── av2_map_infos_val_{newsplit}_gt_trakcs.pkl <-- val gt tracks

```
