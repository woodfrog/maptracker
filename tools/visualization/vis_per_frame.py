import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import argparse     
import mmcv
from mmcv import Config
import os
from mmdet3d.datasets import build_dataset
import torch
import numpy as np
from PIL import Image
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio
from tracking.cmap_utils.match_utils import *


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--out_dir', 
        required=True,
        default='demo',
        help='directory where visualize results will be saved')
    parser.add_argument(
        '--data_path',
        required=True,
        default="",
        help='directory to submission file')
    parser.add_argument(
        '--scene_id',
        type=str, 
        nargs='+',
        default=None,
        help='scene_id to visulize')
    parser.add_argument(
        '--option',
        default="vis-gt",
        help='vis-gt or vis-pred')
    parser.add_argument(
        '--line_opacity',
        default=0.75,
        type=float,
        help='Line simplification tolerance'
    )
    parser.add_argument(
        '--overwrite',
        default=1,
        type=int,
        help='whether to overwrite the existing images'
    )
    parser.add_argument(
        '--dpi',
        default=20,
        type=int,
        help='whether to merge boundary lines'
    )
    
    args = parser.parse_args()

    return args

def save_as_video(image_list, mp4_output_path, scale=None):
    mp4_output_path = mp4_output_path.replace('.gif','.mp4')
    images = [Image.fromarray(img).convert("RGBA") for img in image_list]
    if scale is not None:
        w, h = images[0].size
        images = [img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS) for img in images]
    images = [Image.new('RGBA', images[0].size, (255, 255, 255, 255))] + images
    try:
        imageio.mimsave(mp4_output_path, images,  format='MP4',fps=10)
    except ValueError: # in case the shapes are not the same, have to manually adjust
        resized_images = [img.resize(images[0].size, Image.Resampling.LANCZOS) for img in images]
        print('Size not all the same, manually adjust...')
        imageio.mimsave(mp4_output_path, resized_images,  format='MP4',fps=10)
    print("mp4 saved to : ", mp4_output_path)

def plot_one_frame_results(vectors, id_info, roi_size, scene_dir, args):                
    # setup the figure with car
    plt.figure(figsize=(roi_size[0], roi_size[1]))
    plt.xlim(-roi_size[0] / 2, roi_size[0] / 2)
    plt.ylim(-roi_size[1] / 2, roi_size[1] / 2)
    plt.axis('off')
    plt.autoscale(False)
    car_img = Image.open('resources/car-orange.png')
    plt.imshow(car_img, extent=[-2.2, 2.2, -2, 2])
    
    for label, vecs in vectors.items():
        if label == 0: # ped_crossing
            color = 'b'
            label_text = 'P'
        elif label == 1: # divider
            color = 'r'
            label_text = 'D'
        elif label == 2: # boundary
            color = 'g'
            label_text = 'B'
        
        if len(vecs) == 0:
            continue

        for vec_idx, vec in enumerate(vecs):
            pts = vec[:, :2]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            plt.plot(x, y, 'o-', color=color, linewidth=25, markersize=20, alpha=args.line_opacity)
            vec_id = id_info[label][vec_idx]
            mid_idx = len(x) // 2

            # Put instance id, prevent the text from changing fig size...
            if -roi_size[1]/2 <= y[mid_idx] < -roi_size[1]/2 + 2:
                text_y = y[mid_idx] + 2
            elif roi_size[1]/2 - 2 < y[mid_idx] <= roi_size[1]/2:
                text_y = y[mid_idx] - 2
            else:
                text_y = y[mid_idx]
            
            if -roi_size[0]/2 <= x[mid_idx] < -roi_size[0]/2 + 4:
                text_x = x[mid_idx] + 4
            elif roi_size[0]/2 - 4 < x[mid_idx] <= roi_size[0]/2:
                text_x = x[mid_idx] - 4
            else:
                text_x = x[mid_idx]

            plt.text(text_x, text_y, f'{label_text}{vec_id}', fontsize=80, color=color)
        
    save_path = os.path.join(scene_dir, 'temp.png')
    plt.savefig(save_path, bbox_inches='tight', transparent=False, dpi=args.dpi)
    plt.clf()  
    plt.close()

    viz_image = imageio.imread(save_path)
    return viz_image
    
def vis_pred_data(scene_name, args, pred_results, origin,roi_size):
    
    # get the item index of the scene
    scene_idx = defaultdict(list)
    
    for index in range(len(pred_results)):
        scene_idx[pred_results[index]["scene_name"]].append(index)
        
    index_list = scene_idx[scene_name]
    
    scene_dir = os.path.join(args.out_dir,scene_name)
    os.makedirs(scene_dir,exist_ok=True)

    g2l_id_mapping = dict()
    label_ins_counter = {0:0, 1:0, 2:0}

    all_viz_images = []

    # iterate through each frame of the pred sequence
    for index in index_list:
        vectors = np.array(pred_results[index]["vectors"]).reshape((len(np.array(pred_results[index]["vectors"])), 20, 2))
        # some results are normalized, some not...
        if np.abs(vectors).max() <= 1: 
            vectors = vectors * roi_size + origin
        labels = np.array(pred_results[index]["labels"])
        global_ids = np.array(pred_results[index]["global_ids"])

        per_label_results = defaultdict(list) 

        for ins_idx in range(len(vectors)):
            label = int(labels[ins_idx])
            global_id = int(global_ids[ins_idx])
            if global_id not in g2l_id_mapping:
                local_idx = label_ins_counter[label]
                g2l_id_mapping[global_id] = (label, local_idx)
                label_ins_counter[label] += 1
            else:
                if label == g2l_id_mapping[global_id][0]:
                    local_idx = g2l_id_mapping[global_id][1]
                else: 
                    # label changes for a tracked instance (can happen in our method)
                    # need to update the global id info
                    local_idx = label_ins_counter[label]
                    g2l_id_mapping[global_id] = (label, local_idx)
                    label_ins_counter[label] += 1

            per_label_results[label].append([vectors[ins_idx], global_id, local_idx])

        curr_vectors = defaultdict(list) 
        id_info = dict()
        for label, results in per_label_results.items():
            vec_results = [item[0] for item in results]
            global_ids = [item[1] for item in results]
            local_ids = [item[2] for item in results]

            curr_vectors[label] = np.stack(vec_results, axis=0)
            id_info[label] = {idx:ins_id for idx, ins_id in enumerate(local_ids)}
        
        viz_image = plot_one_frame_results(curr_vectors, id_info, roi_size, scene_dir, args)
        all_viz_images.append(viz_image)
        
    gif_path = os.path.join(scene_dir, 'per_frame_pred.gif')
    save_as_video(all_viz_images, gif_path)
        
def vis_gt_data(scene_name, args, dataset, scene_name2idx, gt_data, origin, roi_size):
    gt_info = gt_data[scene_name]
    gt_info_list = []
    ids_info = []

    scene_dir = os.path.join(args.out_dir,scene_name)
    os.makedirs(scene_dir,exist_ok=True)

    for index, one_idx in enumerate(gt_info["sample_ids"]):
        gt_info_list.append(dataset[one_idx])
        ids_info.append(gt_info["instance_ids"][index])
    scene_len = len(gt_info_list)

    all_viz_images = []
    all_cam_images = {cam_name: [] for cam_name in dataset.samples[0]['cams'].keys()}

    for frame_idx in range(scene_len):
        global_idx = scene_name2idx[scene_name][frame_idx]
        # collect images for each camera
        cams = dataset.samples[global_idx]['cams']
        for cam, info in cams.items():
            img = imageio.imread(info['img_fpath'])
            all_cam_images[cam].append(img)
        # collect vectors for each frame
        curr_vectors = dict()
        for label, vecs in gt_info_list[frame_idx]['vectors'].data.items():
            if len(vecs) > 0:
                curr_vectors[label] = vecs * roi_size + origin
            else:
                curr_vectors[label] = vecs
        
        id_info = ids_info[frame_idx]

        viz_image = plot_one_frame_results(curr_vectors, id_info, roi_size, scene_dir, args)
        all_viz_images.append(viz_image)
    
    gif_path = os.path.join(scene_dir, 'per_frame_gt.gif')
    save_as_video(all_viz_images, gif_path)
    
    for cam_name, image_list in all_cam_images.items():
        gif_path = os.path.join(scene_dir, f'{cam_name}.gif')
        save_as_video(image_list, gif_path, scale=0.3)
    
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    dataset = build_dataset(cfg.match_config)

    scene_name2idx = {}
    scene_name2token = {}
    for idx, sample in enumerate(dataset.samples):
        scene = sample['scene_name']
        if scene not in scene_name2idx:
            scene_name2idx[scene] = []
            scene_name2token[scene] = []
        scene_name2idx[scene].append(idx)

    if args.data_path == "":
        data = {}
    elif args.option == "vis-gt": # visulize GT option
        data = mmcv.load(args.data_path)
    elif args.option == "vis-pred":
        with open(args.data_path,'rb') as fp:
            data = pickle.load(fp)

    all_scene_names = sorted(list(scene_name2idx.keys()))
    scene_info_list = []
    for single_scene_name in all_scene_names:
        scene_info_list.append((single_scene_name, args))

    roi_size = torch.tensor(cfg.roi_size).numpy()
    origin = torch.tensor(cfg.pc_range[:2]).numpy()
    
    for scene_name in all_scene_names:

        if args.scene_id is not None and scene_name not in args.scene_id:
            continue
        scene_dir = os.path.join(args.out_dir,scene_name)
        if os.path.exists(scene_dir) and len(os.listdir(scene_dir)) > 0 and not args.overwrite:
            print(f"Scene {scene_name} already generated, skipping...")
            continue
        os.makedirs(scene_dir,exist_ok=True)
        if args.option == "vis-gt":
            vis_gt_data(scene_name=scene_name, args=args, dataset=dataset, 
                scene_name2idx=scene_name2idx, gt_data=data,origin=origin,roi_size=roi_size)
        elif args.option == "vis-pred":
            vis_pred_data(scene_name=scene_name, args=args, pred_results=data, origin=origin, roi_size=roi_size)

if __name__ == '__main__':
    main()