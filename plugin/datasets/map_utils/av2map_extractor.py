from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path
from shapely.geometry import LineString, box, Polygon
from shapely import ops
import numpy as np
from .utils import split_collections, get_drivable_area_contour, \
        get_ped_crossing_contour, remove_repeated_lines, transform_from, \
        connect_lines, remove_boundary_dividers, remove_repeated_lanesegment, reassign_graph_attribute
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union

from av2.geometry.se3 import SE3
from nuscenes.map_expansion.map_api import NuScenesMapExplorer
import networkx as nx

from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString
from shapely.strtree import STRtree

from shapely.geometry import CAP_STYLE, JOIN_STYLE


class AV2MapExtractor(object):
    """Argoverse 2 map ground-truth extractor.

    Args:
        roi_size (tuple or list): bev range
        id2map (dict): log id to map json path
    """
    def __init__(self, roi_size: Union[Tuple, List], id2map: Dict) -> None:
        self.roi_size = roi_size
        self.id2map = {}

        for log_id, path in id2map.items():
            self.id2map[log_id] = ArgoverseStaticMap.from_json(Path(path))
        
    def generate_nearby_dividers(self,avm, e2g_translation, e2g_rotation,patch):
        def get_path(ls_dict):
            pts_G = nx.DiGraph()
            junction_pts_list = []
            tmp=ls_dict
            for key, value in tmp.items():
                centerline_geom = LineString(value['polyline'].xyz)
                centerline_pts = np.array(centerline_geom.coords).round(3)
                start_pt = centerline_pts[0]
                end_pt = centerline_pts[-1]

                for idx, pts in enumerate(centerline_pts[:-1]):
                    pts_G.add_edge(tuple(centerline_pts[idx]),tuple(centerline_pts[idx+1]))

                valid_incoming_num = 0
                for idx, pred in enumerate(value['predecessors']):
                    if pred in tmp.keys():
                        valid_incoming_num += 1
                        pred_geom = LineString(tmp[pred]['polyline'].xyz)
                        pred_pt = np.array(pred_geom.coords).round(3)[-1]

                        if pred_pt[0] == start_pt[0] and pred_pt[1] == start_pt[1] and pred_pt[2] == start_pt[2]:
                            pass
                        else:
                            pts_G.add_edge(tuple(pred_pt), tuple(start_pt))

                if valid_incoming_num > 1:
                    junction_pts_list.append(tuple(start_pt))
                
                valid_outgoing_num = 0
                for idx, succ in enumerate(value['successors']):
                    if succ in tmp.keys():
                        valid_outgoing_num += 1
                        succ_geom = LineString(tmp[succ]['polyline'].xyz)
                        succ_pt = np.array(succ_geom.coords).round(3)[0]

                        if end_pt[0] == succ_pt[0] and end_pt[1] == succ_pt[1] and end_pt[2] == succ_pt[2]:
                            pass
                        else:
                            pts_G.add_edge(tuple(end_pt), tuple(succ_pt))

                if valid_outgoing_num > 1:
                    junction_pts_list.append(tuple(end_pt))
            
            roots = (v for v, d in pts_G.in_degree() if d == 0)
            roots_list = [v for v, d in pts_G.in_degree() if d == 0]
            
            notroot_list = [v for v in pts_G.nodes if v not in roots_list]
            leaves = [v for v,d in pts_G.out_degree() if d==0]
            ### find path from each root to leaves

            all_paths = []
            for root in roots:
                for leave in leaves:
                    paths = nx.all_simple_paths(pts_G, root, leave)
                    all_paths.extend(paths)

            for single_path in all_paths:
                for single_node in single_path:
                    if single_node in notroot_list:
                        notroot_list.remove(single_node)

            final_centerline_paths = []
            for path in all_paths:
                merged_line = LineString(path)
                # pdb.set_trace()
                merged_line = merged_line.simplify(0.2, preserve_topology=True)
                final_centerline_paths.append(merged_line)

            local_centerline_paths = final_centerline_paths
            return local_centerline_paths
        
        left_lane_dict = {}
        right_lane_dict = {}

        scene_ls_list = avm.get_scenario_lane_segments()
        scene_ls_dict = dict()
        for ls in scene_ls_list:
            scene_ls_dict[ls.id] = dict(
                ls=ls,
                polygon = Polygon(ls.polygon_boundary),
                predecessors=ls.predecessors,
                successors=ls.successors
            )
        
        nearby_ls_dict = dict()
        for key, value in scene_ls_dict.items():
            polygon = value['polygon']
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    nearby_ls_dict[key] = value['ls']

        ls_dict = nearby_ls_dict
        divider_ls_dict = dict()
        for key, value in ls_dict.items():
            if not value.is_intersection:
                divider_ls_dict[key] = value

        left_lane_dict = {}
        right_lane_dict = {}
        for key,value in divider_ls_dict.items():
            if value.left_neighbor_id is not None:
                left_lane_dict[key] = dict(
                    polyline=value.left_lane_boundary,
                    predecessors = value.predecessors,
                    successors = value.successors,
                    left_neighbor_id = value.left_neighbor_id,
                )
            if value.right_neighbor_id is not None:
                right_lane_dict[key] = dict(
                    polyline = value.right_lane_boundary,
                    predecessors = value.predecessors,
                    successors = value.successors,
                    right_neighbor_id = value.right_neighbor_id,
                )

        for key, value in left_lane_dict.items():
            if value['left_neighbor_id'] in right_lane_dict.keys():
                del right_lane_dict[value['left_neighbor_id']]

        for key, value in right_lane_dict.items():
            if value['right_neighbor_id'] in left_lane_dict.keys():
                del left_lane_dict[value['right_neighbor_id']]

        left_lane_dict = remove_repeated_lanesegment(left_lane_dict)
        right_lane_dict = remove_repeated_lanesegment(right_lane_dict)

        left_lane_dict = reassign_graph_attribute(left_lane_dict)
        right_lane_dict = reassign_graph_attribute(right_lane_dict)

        left_paths = get_path(left_lane_dict)
        right_paths = get_path(right_lane_dict)
        local_dividers = left_paths + right_paths

        return local_dividers

    def proc_polygon(self,polygon, ego_SE3_city):
        interiors = []
        exterior_cityframe = np.array(list(polygon.exterior.coords))
        exterior_egoframe = ego_SE3_city.transform_point_cloud(exterior_cityframe)
        for inter in polygon.interiors:
            inter_cityframe = np.array(list(inter.coords))
            inter_egoframe = ego_SE3_city.transform_point_cloud(inter_cityframe)
            interiors.append(inter_egoframe[:,:3])

        new_polygon = Polygon(exterior_egoframe[:,:3], interiors)
        return new_polygon
    
    def proc_line(self,line,ego_SE3_city):
        new_line_pts_cityframe = np.array(list(line.coords))
        new_line_pts_egoframe = ego_SE3_city.transform_point_cloud(new_line_pts_cityframe)
        line = LineString(new_line_pts_egoframe[:,:3]) #TODO
        return line

    def extract_local_divider(self,nearby_dividers, ego_SE3_city, patch_box, patch_angle,patch_size):
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        # pdb.set_trace()
        # final_pgeom = remove_repeated_lines(nearby_dividers)
        line_list = []
        # pdb.set_trace()
        for line in nearby_dividers:
            if line.is_empty:  # Skip lines without nodes.
                continue
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                if new_line.geom_type == 'MultiLineString':
                    for single_line in new_line.geoms:
                        if single_line.is_empty:
                            continue
                        single_line = self.proc_line(single_line,ego_SE3_city)
                        line_list.append(single_line)
                else:
                    new_line = self.proc_line(new_line, ego_SE3_city)
                    line_list.append(new_line)
        centerlines = line_list
        
        poly_centerlines = [line.buffer(0.1,
                    cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre) for line in centerlines]
        index_by_id = dict((id(pt), i) for i, pt in enumerate(poly_centerlines))
        tree = STRtree(poly_centerlines)
        final_pgeom = []
        remain_idx = [i for i in range(len(centerlines))]
        for i, pline in enumerate(poly_centerlines):
            if i not in remain_idx:
                continue
            remain_idx.pop(remain_idx.index(i))

            final_pgeom.append(centerlines[i])
            for o in tree.query(pline):
                o_idx = index_by_id[id(o)]
                if o_idx not in remain_idx:
                    continue
                inter = o.intersection(pline).area
                union = o.union(pline).area
                iou = inter / union
                if iou >= 0.90:
                    remain_idx.pop(remain_idx.index(o_idx))

        # return [np.array(line.coords) for line in final_pgeom]
        final_pgeom = connect_lines(final_pgeom)
        return final_pgeom

    def extract_local_boundary(self,avm, ego_SE3_city, patch_box, patch_angle,patch_size):
        boundary_list = []
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        for da in avm.get_scenario_vector_drivable_areas():
            boundary_list.append(da.xyz)

        polygon_list = []
        for da in boundary_list:
            exterior_coords = da
            interiors = []
        #     polygon = Polygon(exterior_coords, interiors)
            polygon = Polygon(exterior_coords, interiors)
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    if new_polygon.geom_type is 'Polygon':
                        if not new_polygon.is_valid:
                            continue
                        new_polygon = self.proc_polygon(new_polygon,ego_SE3_city)
                        if not new_polygon.is_valid:
                            continue
                    elif new_polygon.geom_type is 'MultiPolygon':
                        polygons = []
                        for single_polygon in new_polygon.geoms:
                            if not single_polygon.is_valid or single_polygon.is_empty:
                                continue
                            new_single_polygon = self.proc_polygon(single_polygon,ego_SE3_city)
                            if not new_single_polygon.is_valid:
                                continue
                            polygons.append(new_single_polygon)
                        if len(polygons) == 0:
                            continue
                        new_polygon = MultiPolygon(polygons)
                        if not new_polygon.is_valid:
                            continue
                    else:
                        raise ValueError('{} is not valid'.format(new_polygon.geom_type))

                    if new_polygon.geom_type is 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        union_segments = ops.unary_union(polygon_list)
        max_x = patch_size[1] / 2
        max_y = patch_size[0] / 2
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

        boundary_lines = []
        for line in results:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        boundary_lines.append(single_line)
                elif line.geom_type == 'LineString':
                    boundary_lines.append(line)
                else:
                    raise NotImplementedError
        return boundary_lines

    def get_scene_dividers(self,avm,patch_box,patch_angle):
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        scene_ls_list = avm.get_scenario_lane_segments()
        # pdb.set_trace()
        scene_ls_dict = dict()
        for ls in scene_ls_list:
            scene_ls_dict[ls.id] = dict(
                ls=ls,
                polygon = Polygon(ls.polygon_boundary),
                predecessors=ls.predecessors,
                successors=ls.successors
            )
        nearby_ls_dict = dict()
        for key, value in scene_ls_dict.items():
            polygon = value['polygon']
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    nearby_ls_dict[key] = value['ls']

        ls_dict = nearby_ls_dict
        divider_ls_dict = dict()
        for key, value in ls_dict.items():
            if not value.is_intersection:
                divider_ls_dict[key] = value

        return divider_ls_dict

    def get_scene_ped_crossings(self,avm,e2g_translation,e2g_rotation,polygon_ped=True):

        g2e_translation = e2g_rotation.T.dot(-e2g_translation)
        g2e_rotation = e2g_rotation.T

        roi_x, roi_y = self.roi_size[:2]
        local_patch = box(-roi_x / 2, -roi_y / 2, roi_x / 2, roi_y / 2)
        ped_crossings = [] 
        for _, pc in avm.vector_pedestrian_crossings.items():
            edge1_xyz = pc.edge1.xyz
            edge2_xyz = pc.edge2.xyz
            ego1_xyz = transform_from(edge1_xyz, g2e_translation, g2e_rotation)
            ego2_xyz = transform_from(edge2_xyz, g2e_translation, g2e_rotation)

            # if True, organize each ped crossing as closed polylines. 
            if polygon_ped:
                vertices = np.concatenate([ego1_xyz, ego2_xyz[::-1, :]])
                p = Polygon(vertices)
                line = get_ped_crossing_contour(p, local_patch)
                if line is not None:
                    if len(line.coords) < 3 or Polygon(line).area < 1:
                        continue
                    ped_crossings.append(line)
            # Otherwise organize each ped crossing as two parallel polylines.
            else:
                line1 = LineString(ego1_xyz)
                line2 = LineString(ego2_xyz)
                line1_local = line1.intersection(local_patch)
                line2_local = line2.intersection(local_patch)

                # take the whole ped cross if all two edges are in roi range
                if not line1_local.is_empty and not line2_local.is_empty:
                    ped_crossings.append(line1_local)
                    ped_crossings.append(line2_local)

        return ped_crossings
    
    def get_map_geom(self,
                     log_id: str, 
                     e2g_translation: NDArray, 
                     e2g_rotation: NDArray,
                     polygon_ped=True) -> Dict[str, List[Union[LineString, Polygon]]]:
        ''' Extract geometries given `log_id` and ego pose.
        
        Args:
            log_id (str): log id
            e2g_translation (array): ego2global translation, shape (3,)
            e2g_rotation (array): ego2global rotation matrix, shape (3, 3)
            polygon_ped: if True, organize each ped crossing as closed polylines. \
                Otherwise organize each ped crossing as two parallel polylines. \
                Default: True
        
        Returns:
            geometries (Dict): extracted geometries by category.
        '''

        avm = self.id2map[log_id]
        
        patch_h = self.roi_size[1]
        patch_w = self.roi_size[0]
        patch_size = (patch_h, patch_w)
        map_pose = e2g_translation[:2]
        rotation = Quaternion._from_matrix(e2g_rotation)
        patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        city_SE2_ego = SE3(e2g_rotation, e2g_translation)
        ego_SE3_city = city_SE2_ego.inverse()
        
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        nearby_dividers = self.generate_nearby_dividers(avm, e2g_translation,e2g_rotation,patch)
        # pdb.set_trace()
        map_anno=dict(
            divider=[],
            ped_crossing=[],
            boundary=[],
            drivable_area=[],
        )
        map_anno['ped_crossing'] = self.get_scene_ped_crossings(avm,e2g_translation,e2g_rotation,polygon_ped=polygon_ped)
        
        map_anno['boundary'] = self.extract_local_boundary(avm, ego_SE3_city, patch_box, patch_angle,patch_size)
        # map_anno['centerline'] = extract_local_centerline(nearby_centerlines, ego_SE3_city, patch_box, patch_angle,patch_size)
        all_dividers = self.extract_local_divider(nearby_dividers, ego_SE3_city, patch_box, patch_angle,patch_size)

        map_anno['divider'] = remove_boundary_dividers(all_dividers,map_anno['boundary'])

        ########
        return map_anno