import numpy as np
from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString, Polygon
from shapely import affinity
import cv2
from PIL import Image, ImageDraw
from numpy.typing import NDArray
from typing import List, Tuple, Union, Dict
import torch

import pdb

@PIPELINES.register_module(force=True)
class RasterizeMap(object):
    """Generate rasterized semantic map and put into 
    `semantic_mask` key.

    Args:
        roi_size (tuple or list): bev range
        canvas_size (tuple or list): bev feature size
        thickness (int): thickness of rasterized lines
        coords_dim (int): dimension of point coordinates
    """

    def __init__(self, 
                 roi_size: Union[Tuple, List], 
                 canvas_size: Union[Tuple, List], 
                 thickness: int, 
                 coords_dim: int,
                 semantic_mask=False,
                 ):

        self.roi_size = roi_size
        self.canvas_size = canvas_size
        self.scale_x = self.canvas_size[0] / self.roi_size[0]
        self.scale_y = self.canvas_size[1] / self.roi_size[1]
        self.thickness = thickness
        self.coords_dim = coords_dim
        self.semantic_mask = semantic_mask

    def line_ego_to_mask(self, 
                         line_ego: LineString, 
                         mask: NDArray, 
                         color: int=1, 
                         thickness: int=3,
                         fill_poly=False
                        ) -> None:
        # """Rasterize a single line to mask.
        # Args:
        #     line_ego (LineString): line
        #     mask (array): semantic mask to paint on
        #     color (int): positive label, default: 1
        #     thickness (int): thickness of rasterized lines, default: 3
        # """

        trans_x = self.canvas_size[0] / 2
        trans_y = self.canvas_size[1] / 2
        line_ego = affinity.scale(line_ego, self.scale_x, self.scale_y, origin=(0, 0))
        line_ego = affinity.affine_transform(line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
        
        coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
        coords = coords.reshape((-1, 2))
        assert len(coords) >= 2
        
        if fill_poly:
            cv2.fillPoly(mask, np.int32([coords]), color=color)
        else:
            cv2.polylines(mask, np.int32([coords]), False, color=color, thickness=thickness)

        
    def polygons_ego_to_mask(self, 
                             polygons: List[Polygon], 
                             color: int=1) -> NDArray:
        # ''' Rasterize a polygon to mask.
        
        # Args:
        #     polygons (list): list of polygons
        #     color (int): positive label, default: 1
        
        # Returns:
        #     mask (array): mask with rasterize polygons
        # '''

        #mask = Image.new("L", size=(self.canvas_size[0], self.canvas_size[1]), color=0) 
        # Image lib api expect size as (w, h)
        trans_x = self.canvas_size[0] / 2
        trans_y = self.canvas_size[1] / 2
        masks = []
        for polygon in polygons:
            mask = Image.new("L", size=(self.canvas_size[0], self.canvas_size[1]), color=0) 
            polygon = affinity.scale(polygon, self.scale_x, self.scale_y, origin=(0, 0))
            polygon = affinity.affine_transform(polygon, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            ext = np.array(polygon.exterior.coords)[:, :2]
            vert_list = [(x, y) for x, y in ext]

            ImageDraw.Draw(mask).polygon(vert_list, outline=1, fill=color)
            masks.append(mask)

        #return np.array(mask, np.uint8)
        return masks
    
    def get_semantic_mask(self, map_geoms: Dict) -> NDArray:
        # ''' Rasterize all map geometries to semantic mask.
        
        # Args:
        #     map_geoms (dict): map geoms by class
        
        # Returns:
        #     semantic_mask (array): semantic mask
        # '''

        num_classes = len(map_geoms)
        if self.semantic_mask:
            semantic_mask = np.zeros((num_classes, self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)
        else:
            instance_masks = []

        for label, geom_list in map_geoms.items():
            if len(geom_list) == 0:
                continue
            if geom_list[0].geom_type == 'LineString':
                for line in geom_list:
                    if self.semantic_mask:
                        fill_poly = True if label == 0 else False
                        self.line_ego_to_mask(line, semantic_mask[label], color=1,
                                            thickness=self.thickness, fill_poly=fill_poly)
                    else:
                        canvas = np.zeros((self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)
                        self.line_ego_to_mask(line, canvas, color=1,
                            thickness=self.thickness, fill_poly=False)
                        instance_masks.append([canvas, label])
            elif geom_list[0].geom_type == 'Polygon':
                # drivable area 
                polygons = []
                for polygon in geom_list:
                    polygons.append(polygon)
                if self.semantic_mask:
                    semantic_mask[label] = self.polygons_ego_to_mask(polygons, color=1)
                else:
                    polygon_masks = self.polygons_ego_to_mask(polygons, color=1)
                    for mask in polygon_masks:
                        instance_masks.append([mask, label])
            else:
                raise ValueError('map geoms must be either LineString or Polygon!')
        
        if self.semantic_mask:
            semantic_mask = np.ascontiguousarray(semantic_mask)
            return semantic_mask
        else:
            return instance_masks

    def __call__(self, input_dict: Dict) -> Dict:
        map_geoms = input_dict['map_geoms'] # {0: List[ped_crossing: LineString], 1: ...}

        semantic_mask = self.get_semantic_mask(map_geoms)
        input_dict['semantic_mask'] = semantic_mask # (num_class, canvas_size[1], canvas_size[0])
        return input_dict
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(roi_size={self.roi_size}, '
        repr_str += f'canvas_size={self.canvas_size}), '
        repr_str += f'thickness={self.thickness}), ' 
        repr_str += f'coords_dim={self.coords_dim})'

        return repr_str


@PIPELINES.register_module(force=True)
class PV_Map(object):
    """Generate rasterized semantic map and put into 
    `semantic_mask` key.

    Args:
        roi_size (tuple or list): bev range
        canvas_size (tuple or list): bev feature size
        thickness (int): thickness of rasterized lines
        coords_dim (int): dimension of point coordinates
    """

    def __init__(self,
                 img_shape: Union[Tuple, List], 
                 feat_down_sample: int,
                 thickness: int, 
                 coords_dim: int,
                 pv_mask=False,
                 num_cams=6,
                 num_coords=2
                 ):

        self.num_cams = num_cams
        self.num_coords = num_coords
        self.img_shape = img_shape
        self.feat_down_sample = feat_down_sample

        self.pv_scale_x = self.img_shape[0] // feat_down_sample
        self.pv_scale_y = self.img_shape[1] // feat_down_sample

        self.thickness = thickness
        self.coords_dim = coords_dim
        self.pv_mask = pv_mask
        
    def perspective(self,cam_coords, proj_mat):
        pix_coords = proj_mat @ cam_coords
        valid_idx = pix_coords[2, :] > 0
        pix_coords = pix_coords[:, valid_idx]
        pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
        pix_coords = pix_coords.transpose(1, 0)
        return pix_coords

    @staticmethod
    def get_valid_pix_coords(pix_coords):
        valid_idx = pix_coords[:, 2] > 0
        pix_coords = pix_coords[valid_idx, :]
        pix_coords = pix_coords[:, :2] / (pix_coords[:, 2:3] + 1e-7)
        return pix_coords

    def line_ego_to_pvmask(self,
                          line_ego, 
                          mask, 
                          lidar2feat,
                          color=1, 
                          thickness=1):

        distances = np.linspace(0, line_ego.length, 200)
        coords = np.array([np.array(line_ego.interpolate(distance).coords) for distance in distances]).reshape(-1, self.num_coords)
        if coords.shape[1] == 2:
            coords = np.concatenate((coords,np.zeros((coords.shape[0],1))),axis=1)
        
        pts_num = coords.shape[0]
        ones = np.ones((pts_num,1))
        lidar_coords = np.concatenate([coords,ones], axis=1).transpose(1,0)
        pix_coords = self.perspective(lidar_coords, lidar2feat) // self.feat_down_sample
        cv2.polylines(mask, np.int32([pix_coords]), False, color=color, thickness=thickness)
    
    def lines_ego_to_pv(self, lines_ego, pv_mask, ego2imgs, color=1, thickness=1):
        lines_coord = []
        for line_ego in lines_ego:
            distances = np.linspace(0, line_ego.length, 100)
            coords = np.array([np.array(line_ego.interpolate(distance).coords) for distance in distances]).reshape(-1, self.num_coords)
            if coords.shape[1] == 2:
                coords = np.concatenate((coords,np.zeros((coords.shape[0],1))),axis=1)
            pts_num = coords.shape[0]
            ones = np.ones((pts_num,1))
            lidar_coords = np.concatenate([coords,ones], axis=1)
            lines_coord.append(lidar_coords)
        lines_coord = torch.tensor(np.stack(lines_coord, axis=0))
        for cam_idx in range(len(ego2imgs)):
            ego2img_i = torch.tensor(ego2imgs[cam_idx])
            pers_lines_coord = torch.einsum('lk,ijk->ijl', ego2img_i, lines_coord)
            valid_lines_coord = [self.get_valid_pix_coords(pers_coord) for pers_coord in pers_lines_coord]
            valid_lines_coord = [x // self.feat_down_sample for x in valid_lines_coord if len(x) > 0]
            lines_to_draw = [x.numpy().astype(np.int32) for x in valid_lines_coord]
            cv2.polylines(pv_mask[cam_idx], lines_to_draw, False, color=color, thickness=thickness)
    
    def get_pvmask_old(self,map_geoms: Dict,ego2img: List, img_filenames: List) -> NDArray:
        # ''' Rasterize all map geometries to semantic mask.
    
        # Args:
        #     map_geoms (dict): map geoms by class
    
        # Returns:
        #     semantic_mask (array): semantic mask
        # '''
        num_classes = len(map_geoms)
        if self.pv_mask:
            gt_pv_mask = np.zeros((self.num_cams, num_classes, self.pv_scale_x, self.pv_scale_y), dtype=np.uint8)
        else:
            instance_masks = []

        for label, geom_list in map_geoms.items():
            if len(geom_list) == 0:
                continue
            if geom_list[0].geom_type == 'LineString':
                for line in geom_list:
                    for cam_index in range(self.num_cams):
                        self.line_ego_to_pvmask(line,gt_pv_mask[cam_index][label],ego2img[cam_index],color=1,thickness=self.thickness)
        if self.pv_mask:
             gt_pv_mask = np.ascontiguousarray(gt_pv_mask)
            ## Visualize to double-check the pv seg is correct
             #self.visualize_all_pv_masks(gt_pv_mask, img_filenames)
             #import pdb; pdb.set_trace()
             return gt_pv_mask
        else:
            return instance_masks

    def get_pvmask(self, map_geoms: Dict,ego2img: List, img_filenames: List) -> NDArray:
        # ''' Rasterize all map geometries to semantic mask.
        
        # Args:
        #     map_geoms (dict): map geoms by class
        
        # Returns:
        #     semantic_mask (array): semantic mask
        # '''
        num_classes = len(map_geoms)
        if self.pv_mask:
            gt_pv_mask = np.zeros((num_classes, self.num_cams, self.pv_scale_x, self.pv_scale_y), dtype=np.uint8)
        else:
            instance_masks = []

        for label, geom_list in map_geoms.items():
            if len(geom_list) == 0:
                continue
            self.lines_ego_to_pv(geom_list, gt_pv_mask[label], ego2img, color=1, thickness=self.thickness)

        gt_pv_mask = gt_pv_mask.transpose(1, 0, 2, 3)
        if self.pv_mask:
            gt_pv_mask = np.ascontiguousarray(gt_pv_mask)
            ## Visualize to double-check the pv seg is correct
            #self.visualize_all_pv_masks(gt_pv_mask, img_filenames)
            #import pdb; pdb.set_trace()
            return gt_pv_mask
        else:
            return instance_masks

    def __call__(self, input_dict: Dict) -> Dict:
        map_geoms = input_dict['map_geoms'] # {0: List[ped_crossing: LineString], 1: ...}
        pv_mask = self.get_pvmask(map_geoms, input_dict['ego2img'], input_dict['img_filenames'])
        input_dict['pv_mask'] =  pv_mask # (num_class, canvas_size[1], canvas_size[0])
        return input_dict
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(roi_size={self.roi_size}, '
        repr_str += f'canvas_size={self.canvas_size}), '
        repr_str += f'thickness={self.thickness}), ' 
        repr_str += f'coords_dim={self.coords_dim})'

        return repr_str
    
    def visualize_all_pv_masks(self, gt_pv_mask, img_filenames):
        for cam_id in range(gt_pv_mask.shape[0]):
            viz_img = self._visualize_pv_mask(gt_pv_mask[cam_id])
            viz_img = viz_img.transpose(1, 2, 0)
            out_path = './check_pv_seg/viz_{}.png'.format(cam_id)
            out_raw_path = './check_pv_seg/viz_raw_{}.png'.format(cam_id)
            filepath = img_filenames[cam_id]
            pv_img = cv2.imread(filepath)
            #pv_img = cv2.resize(pv_img, (800, 480))
            #viz_mask = cv2.resize(viz_img, (800, 480))
            pv_img = cv2.resize(pv_img, (608, 608))
            viz_mask = cv2.resize(viz_img, (608, 608))
            mask = (viz_mask == 255).all(-1)[..., None]
            viz_img = pv_img * mask + viz_mask * (1-mask)
            cv2.imwrite(out_path, viz_img)
            cv2.imwrite(out_raw_path, pv_img)
    
    def _visualize_pv_mask(self, pv_mask):
        COLOR_MAPS_BGR = {
            # bgr colors
            1: (0, 0, 255),
            2: (0, 255, 0),
            0: (255, 0, 0),
        }
        num_classes, h, w = pv_mask.shape
        viz_img = np.ones((num_classes, h, w), dtype=np.uint8) * 255
        for label in range(num_classes):
            valid = (pv_mask[label] == 1)
            viz_img[:, valid] = np.array(COLOR_MAPS_BGR[label]).reshape(3, 1)

        return viz_img