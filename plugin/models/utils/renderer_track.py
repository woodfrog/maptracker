import os.path as osp
import os
#import av2.geometry.interpolate as interp_utils
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def remove_nan_values(uv):
    is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
    is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
    is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

    uv_valid = uv[is_uv_valid]
    return uv_valid

def points_ego2img(pts_ego, extrinsics, intrinsics):
    pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
    pts_cam_4d = extrinsics @ pts_ego_4d.T
    
    uv = (intrinsics @ pts_cam_4d[:3, :]).T
    uv = remove_nan_values(uv)
    depth = uv[:, 2]
    uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)

    return uv, depth

def draw_polyline_ego_on_img(polyline_ego, img_bgr, extrinsics, intrinsics, color_bgr, thickness):
    if polyline_ego.shape[1] == 2:
        zeros = np.zeros((polyline_ego.shape[0], 1))
        polyline_ego = np.concatenate([polyline_ego, zeros], axis=1)

    polyline_ego = interp_utils.interp_arc(t=500, points=polyline_ego)
    
    uv, depth = points_ego2img(polyline_ego, extrinsics, intrinsics)

    h, w, c = img_bgr.shape

    is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < w - 1)
    is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < h - 1)
    is_valid_z = depth > 0
    is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])

    if is_valid_points.sum() == 0:
        return
    
    uv = np.round(uv[is_valid_points]).astype(np.int32)

    draw_visible_polyline_cv2(
        copy.deepcopy(uv),
        valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
        image=img_bgr,
        color=color_bgr,
        thickness_px=thickness,
    )

def draw_visible_polyline_cv2(line, valid_pts_bool, image, color, thickness_px):
    """Draw a polyline onto an image using given line segments.

    Args:
        line: Array of shape (K, 2) representing the coordinates of line.
        valid_pts_bool: Array of shape (K,) representing which polyline coordinates are valid for rendering.
            For example, if the coordinate is occluded, a user might specify that it is invalid.
            Line segments touching an invalid vertex will not be rendered.
        image: Array of shape (H, W, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        thickness_px: thickness (in pixels) to use when rendering the polyline.
    """
    line = np.round(line).astype(int)  # type: ignore
    for i in range(len(line) - 1):

        if (not valid_pts_bool[i]) or (not valid_pts_bool[i + 1]):
            continue

        x1 = line[i][0]
        y1 = line[i][1]
        x2 = line[i + 1][0]
        y2 = line[i + 1][1]

        # Use anti-aliasing (AA) for curves
        image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness_px, lineType=cv2.LINE_AA)


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

CAM_NAMES_AV2 = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left',
    ]
CAM_NAMES_NUSC = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',]

class Renderer(object):
    """Render map elements on image views.

    Args:
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        dataset (str): 'av2' or 'nusc'
    """

    def __init__(self, cat2id, roi_size, dataset='av2'):
        self.roi_size = roi_size
        self.cat2id = cat2id
        self.id2cat = {v: k for k, v in cat2id.items()}
        if dataset == 'av2':
            self.cam_names = CAM_NAMES_AV2
        else:
            self.cam_names = CAM_NAMES_NUSC

    def render_bev_from_vectors(self, vectors, labels, out_path, id_info=None, score_info=None):
        '''Render bev segmentation using vectorized map elements.
        
        Args:
            vectors (list): list of vectorized map elements.
            labels (list): list of labels of map elements.
            out_dir (str): output directory
        '''
        if id_info is not None:
            assert len(vectors) == len(id_info)
        if score_info is not None:
            assert len(vectors) == len(score_info)

        car_img = Image.open('resources/car.png')

        plt.figure(figsize=(self.roi_size[0], self.roi_size[1]))
        plt.xlim(-self.roi_size[0] / 2, self.roi_size[0] / 2)
        plt.ylim(-self.roi_size[1] / 2, self.roi_size[1] / 2)
        plt.axis('off')
        plt.imshow(car_img, extent=[-2.5, 2.5, -2.0, 2.0])

        for idx in range(len(labels)):
            cat = self.id2cat[labels[idx].item()]
            color = COLOR_MAPS_PLT[cat]
            vector = vectors[idx].detach().cpu().numpy()
            pts = vector[:, :2]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], angles='xy', color=color,
            #     scale_units='xy', scale=1)
            # for i in range(len(x)):
            plt.plot(x, y, 'o-', color=color, linewidth=20, markersize=50)
            if id_info is not None:
                vec_id = int(id_info[idx])
                mid_idx = len(x) // 2
                if vec_id == -1:
                    plt.text(x[mid_idx], y[mid_idx], 'FP', fontsize=100, color=color)
                else:
                    plt.text(x[mid_idx], y[mid_idx], '{}'.format(vec_id), fontsize=100, color=color)
            
            if score_info is not None:
                mid_idx = len(x) // 2
                plt.text(x[mid_idx]-1, y[mid_idx]+2, '{:.2f}'.format(score_info[idx]), fontsize=100, color='purple')
                
                    
        plt.savefig(out_path, bbox_inches='tight', dpi=40)
        plt.close()

    def render_bev_from_mask(self, semantic_mask, out_path):
        '''Render bev segmentation from semantic_mask.
        
        Args:
            semantic_mask (array): semantic mask.
            out_dir (str): output directory
        '''

        c, h, w = semantic_mask.shape
        bev_img = np.ones((3, h, w), dtype=np.uint8) * 255
        if 'drivable_area' in self.cat2id:
            drivable_area_mask = semantic_mask[self.cat2id['drivable_area']]
            bev_img[:, drivable_area_mask == 1] = \
                    np.array(COLOR_MAPS_BGR['drivable_area']).reshape(3, 1)
        
        # NOTE: the semantic mask has been changed into instance masks for our use
        for label in range(c):
            cat = self.id2cat[label]
            if cat == 'drivable_area':
                continue
            valid = semantic_mask[label] == 1
            bev_img[:, valid] = np.array(COLOR_MAPS_BGR[cat]).reshape(3, 1)

        #for label in range(c):
        #    cat = self.id2cat[label]
        #    if cat == 'drivable_area':
        #        continue
        #    mask = semantic_mask[label]
        #    valid = mask == 1
        #    bev_img[:, valid] = np.array(COLOR_MAPS_BGR[cat]).reshape(3, 1)
        
        cv2.imwrite(out_path, bev_img.transpose((1, 2, 0)))
        