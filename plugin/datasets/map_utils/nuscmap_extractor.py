from shapely.geometry import LineString, box, Polygon
from shapely import ops, strtree

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from .utils import split_collections, get_drivable_area_contour, get_ped_crossing_contour
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union

from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box, MultiLineString
from shapely import affinity, ops
import networkx as nx


class NuscMapExtractor(object):
    """NuScenes map ground-truth extractor.

    Args:
        data_root (str): path to nuScenes dataset
        roi_size (tuple or list): bev range
    """
    def __init__(self, data_root: str, roi_size: Union[List, Tuple]) -> None:
        self.roi_size = roi_size
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=data_root, map_name=loc)
            self.map_explorer[loc] = CNuScenesMapExplorer(self.nusc_maps[loc])
    
    def get_map_geom(self, 
                     location: str, 
                     e2g_translation: Union[List, NDArray],
                     e2g_rotation: Union[List, NDArray]) -> Dict[str, List[Union[LineString, Polygon]]]:
        # Borrowed from MapTR's codebase to make sure data are the same
        # (center_x, center_y, len_y, len_x) in nuscenes format
        patch_size_ego_coord = (self.roi_size[1], self.roi_size[0])
        patch_size_lidar_coord = (self.roi_size[0], self.roi_size[1])

        vector_map_maptr = VectorizedLocalMap(self.nusc_maps[location], self.map_explorer[location],
                                patch_size_lidar_coord, patch_size_ego_coord, map_classes=['divider','ped_crossing','boundary'])
        map_annos = vector_map_maptr.gen_vectorized_samples(e2g_translation, e2g_rotation)
        
        return dict(
            divider=map_annos['divider'], # List[LineString]
            ped_crossing=map_annos['ped_crossing'], # List[LineString]
            boundary=map_annos['boundary'], # List[LineString]
            drivable_area=[], # List[Polygon],
        )


class VectorizedLocalMap(object):
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'others': -1
    }
    def __init__(self,
                 nusc_map,
                 map_explorer,
                 patch_size,
                 roi_size,
                 map_classes=['divider','ped_crossing','boundary','centerline'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 centerline_classes=['lane_connector','lane'],
                 use_simplify=True,
                 ):
        super().__init__()
        self.nusc_map = nusc_map
        self.map_explorer = map_explorer
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.centerline_classes = centerline_classes
        self.patch_size = patch_size
        self.roi_size = roi_size
        self.local_patch = box(-self.roi_size[0] / 2, -self.roi_size[1] / 2, 
                self.roi_size[0] / 2, self.roi_size[1] / 2)


    def gen_vectorized_samples(self, lidar2global_translation, lidar2global_rotation):
        '''
        use lidar2global to get gt map layers
        '''
        
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        # import ipdb;ipdb.set_trace()
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        map_dict = {'divider':[],'ped_crossing':[],'boundary':[],'centerline':[]}
        vectors = []

        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes)
                line_instances_dict = self.line_geoms_to_instances(line_geom)     
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        instance = affinity.rotate(instance, -90, origin=(0, 0), use_radians=False)
                        map_dict[vec_class].append(instance)
                        # vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes)
                ped_instance_list = ped_geom['ped_crossing']
                #ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    # vectors.append((instance, self.CLASS2LABEL.get('ped_crossing', -1)))
                    instance = affinity.rotate(instance, -90, origin=(0, 0), use_radians=False)
                    map_dict[vec_class].append(instance)
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for instance in poly_bound_list:
                    # import ipdb;ipdb.set_trace()
                    instance = affinity.rotate(instance, -90, origin=(0, 0), use_radians=False)
                    map_dict[vec_class].append(instance)
                    # vectors.append((contour, self.CLASS2LABEL.get('contours', -1)))
            elif vec_class =='centerline':
                centerline_geom = self.get_centerline_geom(patch_box, patch_angle, self.centerline_classes)
                centerline_list = self.centerline_geoms_to_instances(centerline_geom)
                for instance in centerline_list:
                    instance = affinity.rotate(instance, -90, origin=(0, 0), use_radians=False)
                    map_dict[vec_class].append(instance)
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')
        return map_dict

    def get_centerline_geom(self, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.centerline_classes:
                return_token = False
                layer_centerline_dict = self.map_explorer._get_centerline(
                patch_box, patch_angle, layer_name, return_token=return_token)
                if len(layer_centerline_dict.keys()) == 0:
                    continue
                # import ipdb;ipdb.set_trace()
                map_geom.update(layer_centerline_dict)
        return map_geom

    def get_map_geom(self, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(patch_box, patch_angle, layer_name)
                # map_geom.append((layer_name, geoms))
                map_geom[layer_name] = geoms
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(patch_box, patch_angle, layer_name)
                # map_geom.append((layer_name, geoms))
                map_geom[layer_name] = geoms
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line_stmmapnet(patch_box, patch_angle)
                # map_geom.append((layer_name, geoms))
                map_geom[layer_name] = geoms
        return map_geom

    def get_divider_line(self,patch_box,patch_angle,layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == 'traffic_light':
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_explorer.map_api, layer_name)
        for record in records:
            line = self.map_explorer.map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list

    def get_contour_line(self,patch_box,patch_angle,layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_explorer.map_api, layer_name)

        polygon_list = []
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_explorer.map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_explorer.map_api.extract_polygon(record['polygon_token'])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list


    def get_ped_crossing_line(self, patch_box, patch_angle):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)
        polygon_list = []
        records = getattr(self.map_explorer.map_api, 'ped_crossing')
        # records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer.map_api.extract_polygon(record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list
    
    def _union_ped_stmmapnet(self, ped_geoms: List[Polygon]) -> List[Polygon]:
        ''' merge close ped crossings.
        
        Args:
            ped_geoms (list): list of Polygon
        
        Returns:
            union_ped_geoms (Dict): merged ped crossings 
        '''

        ped_geoms = sorted(ped_geoms, key=lambda x:x.area, reverse=True)

        def get_rec_direction(geom):
            rect = geom.minimum_rotated_rectangle
            rect_v_p = np.array(rect.exterior.coords)[:3]
            rect_v = rect_v_p[1:]-rect_v_p[:-1]
            v_len = np.linalg.norm(rect_v, axis=-1)
            longest_v_i = v_len.argmax()

            return rect_v[longest_v_i], v_len[longest_v_i]

        tree = strtree.STRtree(ped_geoms)
        index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

        final_pgeom = []
        remain_idx = [i for i in range(len(ped_geoms))]
        for i, pgeom in enumerate(ped_geoms):

            if i not in remain_idx:
                continue
            # update
            remain_idx.pop(remain_idx.index(i))
            pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
            final_pgeom.append(pgeom)

            intersect_pgeom = tree.query(pgeom)
            intersect_pgeom = sorted(intersect_pgeom, key=lambda x:x.area, reverse=True)
            for o in intersect_pgeom:
                o_idx = index_by_id[id(o)]
                if o_idx not in remain_idx:
                    continue

                o_v, o_v_norm = get_rec_direction(o)
                cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)

                o_pgeom_union = o.union(pgeom)
                ch_union = o_pgeom_union.convex_hull
                ch_area_ratio = o_pgeom_union.area / ch_union.area

                # add an extra criterion for merging here to handle patch-boundary-case
                if 1 - np.abs(cos) < 0.01 and ch_area_ratio > 0.8:  # theta < 8 degrees.
                    final_pgeom[-1] =\
                        final_pgeom[-1].union(o)
                    # update
                    remain_idx.pop(remain_idx.index(o_idx))
        
        final_pgeom = self._handle_small_peds(final_pgeom)

        results = []
        for p in final_pgeom:
            results.extend(split_collections(p))
        return results
    
    def _handle_small_peds(self, ped_geoms):
        def get_two_rec_directions(geom):
            rect = geom.minimum_rotated_rectangle
            rect_v_p = np.array(rect.exterior.coords)[:3]
            rect_v = rect_v_p[1:]-rect_v_p[:-1]
            v_len = np.linalg.norm(rect_v, axis=-1)
            return rect_v, v_len

        tree = strtree.STRtree(ped_geoms)
        index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

        final_pgeom = []
        remain_idx = [i for i in range(len(ped_geoms))]

        for i, pgeom in enumerate(ped_geoms):
            if i not in remain_idx:
                continue
            # update
            remain_idx.pop(remain_idx.index(i))
            final_pgeom.append(pgeom)

            pgeom_v, pgeom_v_norm = get_two_rec_directions(pgeom)
            
            intersect_pgeom = tree.query(pgeom)
            intersect_pgeom = sorted(intersect_pgeom, key=lambda x:x.area, reverse=True)
            for o in intersect_pgeom:
                o_idx = index_by_id[id(o)]
                if o_idx not in remain_idx:
                    continue

                if o.area >= pgeom.area:
                    continue

                o_pgeom_union = o.union(pgeom)
                o_v, o_v_norm = get_two_rec_directions(o_pgeom_union)

                ch_union = o_pgeom_union.convex_hull
                ch_area_ratio = o_pgeom_union.area / ch_union.area
                #mrr_union = o_pgeom_union.minimum_rotated_rectangle
                #mrr_area_ratio = o_pgeom_union.area / mrr_union.area

                cos_00 = pgeom_v[0].dot(o_v[0])/(pgeom_v_norm[0]*o_v_norm[0])
                cos_01 = pgeom_v[0].dot(o_v[1])/(pgeom_v_norm[0]*o_v_norm[1])
                cos_10 = pgeom_v[1].dot(o_v[0])/(pgeom_v_norm[1]*o_v_norm[0])
                cos_11 = pgeom_v[1].dot(o_v[1])/(pgeom_v_norm[1]*o_v_norm[1])
                cos_checks = np.array([(1 - np.abs(cos) < 0.001) for cos in [cos_00, cos_01, cos_10, cos_11]])
                # add an extra criterion for merging here to handle patch-boundary-case

                if cos_checks.sum() == 2 and ch_area_ratio > 0.8:
                    final_pgeom[-1] =\
                        final_pgeom[-1].union(o)
                    # update
                    remain_idx.pop(remain_idx.index(o_idx))

        return final_pgeom


    def get_ped_crossing_line_stmmapnet(self, patch_box, patch_angle):
        # get ped crossings
        ped_crossings = []
        ped = self.map_explorer._get_layer_polygon(
                    patch_box, patch_angle, 'ped_crossing')
                
        for p in ped:
            ped_crossings += split_collections(p)
        # some ped crossings are split into several small parts
        # we need to merge them
        ped_crossings = self._union_ped_stmmapnet(ped_crossings)

        # NOTE: clean-up noisy ped-crossing instances (for our cleaned training data only, maybe need to still
        # use the original version when evaluation...)
        # 1). filter too small ped_crossing merging results 
        #areas = [p.area for p in ped_crossings]
        #print('Ped areas\n', areas)
        updated_ped_crossings = []
        for p_idx, p in enumerate(ped_crossings):
            area = p.area
            if area < 1:
                continue
            elif area < 20:
                covered = False
                for other_idx, p_other in enumerate(ped_crossings):
                    if other_idx != p_idx and p.covered_by(p_other):
                        covered = True
                        break
                if not covered:
                    updated_ped_crossings.append(p)
            else:
                updated_ped_crossings.append(p)

        ped_crossing_lines = []
        for p in updated_ped_crossings:
            # extract exteriors to get a closed polyline                        
            line = get_ped_crossing_contour(p, self.local_patch)
            if line is not None:
                ped_crossing_lines.append(line)
    
        return ped_crossing_lines

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom.items():
            one_type_instances = self._one_type_line_geom_to_instances(a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []
        
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def ped_poly_geoms_to_instances(self, ped_geom):
        # ped = ped_geom[0][1]
        # import ipdb;ipdb.set_trace()
        ped = ped_geom['ped_crossing']
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)


    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom['road_segment']
        lanes = polygon_geom['lane']
        # import ipdb;ipdb.set_trace()
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def centerline_geoms_to_instances(self,geoms_dict):
        centerline_geoms_list,pts_G = self.union_centerline(geoms_dict)
        # vectors_dict = self.centerline_geoms2vec(centerline_geoms_list)
        # import ipdb;ipdb.set_trace()
        return self._one_type_line_geom_to_instances(centerline_geoms_list)


    def centerline_geoms2vec(self, centerline_geoms_list):
        vector_dict = {}
        # import ipdb;ipdb.set_trace()
        # centerline_geoms_list = [line.simplify(0.2, preserve_topology=True) \
        #                         for line in centerline_geoms_list]
        vectors = self._geom_to_vectors(
            centerline_geoms_list)
        vector_dict.update({'centerline': ('centerline', vectors)})
        return vector_dict

    def union_centerline(self, centerline_geoms):
        # import ipdb;ipdb.set_trace()
        pts_G = nx.DiGraph()
        junction_pts_list = []
        for key, value in centerline_geoms.items():
            centerline_geom = value['centerline']
            if centerline_geom.geom_type == 'MultiLineString':
                start_pt = np.array(centerline_geom.geoms[0].coords).round(3)[0]
                end_pt = np.array(centerline_geom.geoms[-1].coords).round(3)[-1]
                for single_geom in centerline_geom.geoms:
                    single_geom_pts = np.array(single_geom.coords).round(3)
                    for idx, pt in enumerate(single_geom_pts[:-1]):
                        pts_G.add_edge(tuple(single_geom_pts[idx]),tuple(single_geom_pts[idx+1]))
            elif centerline_geom.geom_type == 'LineString':
                centerline_pts = np.array(centerline_geom.coords).round(3)
                start_pt = centerline_pts[0]
                end_pt = centerline_pts[-1]
                for idx, pts in enumerate(centerline_pts[:-1]):
                    pts_G.add_edge(tuple(centerline_pts[idx]),tuple(centerline_pts[idx+1]))
            else:
                raise NotImplementedError
            valid_incoming_num = 0
            for idx, pred in enumerate(value['incoming_tokens']):
                if pred in centerline_geoms.keys():
                    valid_incoming_num += 1
                    pred_geom = centerline_geoms[pred]['centerline']
                    if pred_geom.geom_type == 'MultiLineString':
                        pred_pt = np.array(pred_geom.geoms[-1].coords).round(3)[-1]
        #                 if pred_pt != centerline_pts[0]:
                        pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
                    else:
                        pred_pt = np.array(pred_geom.coords).round(3)[-1]
                        pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
            if valid_incoming_num > 1:
                junction_pts_list.append(tuple(start_pt))
            
            valid_outgoing_num = 0
            for idx, succ in enumerate(value['outgoing_tokens']):
                if succ in centerline_geoms.keys():
                    valid_outgoing_num += 1
                    succ_geom = centerline_geoms[succ]['centerline']
                    if succ_geom.geom_type == 'MultiLineString':
                        succ_pt = np.array(succ_geom.geoms[0].coords).round(3)[0]
        #                 if pred_pt != centerline_pts[0]:
                        pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
                    else:
                        succ_pt = np.array(succ_geom.coords).round(3)[0]
                        pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
            if valid_outgoing_num > 1:
                junction_pts_list.append(tuple(end_pt))

        roots = (v for v, d in pts_G.in_degree() if d == 0)
        leaves = [v for v, d in pts_G.out_degree() if d == 0]
        all_paths = []
        for root in roots:
            paths = nx.all_simple_paths(pts_G, root, leaves)
            all_paths.extend(paths)

        final_centerline_paths = []
        for path in all_paths:
            merged_line = LineString(path)
            merged_line = merged_line.simplify(0.2, preserve_topology=True)
            final_centerline_paths.append(merged_line)
        return final_centerline_paths, pts_G


class CNuScenesMapExplorer(NuScenesMapExplorer):
    def __ini__(self, *args, **kwargs):
        super(self, CNuScenesMapExplorer).__init__(*args, **kwargs)

    def _get_centerline(self,
                           patch_box: Tuple[float, float, float, float],
                           patch_angle: float,
                           layer_name: str,
                           return_token: bool = False) -> dict:
        """
         Retrieve the centerline of a particular layer within the specified patch.
         :param patch_box: Patch box defined as [x_center, y_center, height, width].
         :param patch_angle: Patch orientation in degrees.
         :param layer_name: name of map layer to be extracted.
         :return: dict(token:record_dict, token:record_dict,...)
         """
        if layer_name not in ['lane','lane_connector']:
            raise ValueError('{} is not a centerline layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_api, layer_name)

        centerline_dict = dict()
        for record in records:
            if record['polygon_token'] is None:
                # import ipdb
                # ipdb.set_trace()
                continue
            polygon = self.map_api.extract_polygon(record['polygon_token'])

            # if polygon.intersects(patch) or polygon.within(patch):
            #     if not polygon.is_valid:
            #         print('within: {}, intersect: {}'.format(polygon.within(patch), polygon.intersects(patch)))
            #         print('polygon token {} is_valid: {}'.format(record['polygon_token'], polygon.is_valid))

            # polygon = polygon.buffer(0)

            if polygon.is_valid:
                # if within or intersect :

                new_polygon = polygon.intersection(patch)
                # new_polygon = polygon

                if not new_polygon.is_empty:
                    centerline = self.map_api.discretize_lanes(
                            record, 0.5)
                    centerline = list(self.map_api.discretize_lanes([record['token']], 0.5).values())[0]
                    centerline = LineString(np.array(centerline)[:,:2].round(3))
                    if centerline.is_empty:
                        continue
                    centerline = centerline.intersection(patch)
                    if not centerline.is_empty:
                        centerline = \
                            to_patch_coord(centerline, patch_angle, patch_x, patch_y)
                        
                        # centerline.coords = np.array(centerline.coords).round(3)
                        # if centerline.geom_type != 'LineString':
                            # import ipdb;ipdb.set_trace()
                        record_dict = dict(
                            centerline=centerline,
                            token=record['token'],
                            incoming_tokens=self.map_api.get_incoming_lane_ids(record['token']),
                            outgoing_tokens=self.map_api.get_outgoing_lane_ids(record['token']),
                        )
                        centerline_dict.update({record['token']: record_dict})
        return centerline_dict

def to_patch_coord(new_polygon, patch_angle, patch_x, patch_y):
    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                  origin=(patch_x, patch_y), use_radians=False)
    new_polygon = affinity.affine_transform(new_polygon,
                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
    return new_polygon