import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import argparse     
import mmcv
from mmcv import Config
import matplotlib.transforms as transforms
from mmdet3d.datasets import build_dataset
import cv2
import torch
import numpy as np
from PIL import Image
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from scipy.spatial import ConvexHull
from PIL import Image
import cv2
import imageio
import math
from tracking.cmap_utils.match_utils import *


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--out_dir',
        required=True,
        default="",
        help='')
    parser.add_argument(
        '--data_path',
        required=True,
        default="",
        help='Directory to submission file')
    parser.add_argument(
        '--scene_id',
        type=str, 
        nargs='+',
        default=None,
        help='Specify the scene_id to visulize')
    parser.add_argument(
        '--option',
        required=True,
        default="vis-pred",
        help='vis-pred, vis-gt')
    parser.add_argument(
        '--simplify',
        default=0.5,
        type=float,
        help='Line simplification tolerance'
    )
    parser.add_argument(
        '--line_opacity',
        default=0.75,
        type=float,
        help='Line opacity'
    )
    parser.add_argument(
        '--overwrite',
        default=1,
        type=int,
        help='Whether to overwrite the existing visualization files'
    )
    parser.add_argument(
        '--per_frame_result',
        default=1,
        type=int,
        help='Whether to visualize per frame result'
    )
    parser.add_argument(
        '--dpi',
        default=20,
        type=int,
        help='DPI of the output image'
    )
    parser.add_argument(
        '--transparent',
        default=False,
        action='store_true',
        help='Whether to use transparent background'
    )
    
    args = parser.parse_args()

    return args

def combine_images_with_labels(image_paths, labels, output_path, font_scale=0.5, font_color=(0, 0, 0)):
    # Load images
    images = [cv2.imread(path) for path in image_paths]
    
    # Determine the maximum dimensions
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    
    # Create a blank white canvas to hold the 2x2 grid of images
    final_image = np.ones((max_height * 1, max_width * 2, 3), dtype=np.uint8) * 255
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i, img in enumerate(images):
        # Resize image if necessary
        img = cv2.resize(img, (max_width, max_height))
        
        # Calculate position for each image
        x_offset = (i % 2) * max_width
        y_offset = (i // 2) * max_height
        
        # Place image in the canvas
        final_image[y_offset:y_offset+max_height, x_offset:x_offset+max_width] = img
        
        # Add label
        cv2.putText(final_image, labels[i], (x_offset + 5, y_offset + 15), font, font_scale, font_color, 1, cv2.LINE_AA)
    
    # Save the final image
    cv2.imwrite(output_path, final_image)


def merge_corssing(polylines):
    convex_hull_polygon = find_largest_convex_hull(polylines)
    return convex_hull_polygon


def find_largest_convex_hull(polylines):
    # Merge all points from the polylines into a single collection
    all_points = []
    for polyline in polylines:
        all_points.extend(list(polyline.coords))
    
    # Convert the points to a NumPy array for processing with scipy
    points_array = np.array(all_points)
    
    # Compute the convex hull using scipy
    hull = ConvexHull(points_array)
    
    # Extract the vertices of the convex hull
    hull_points = points_array[hull.vertices]
    
    # Create a shapely Polygon object representing the convex hull
    convex_hull_polygon = LineString(hull_points).convex_hull
    
    return convex_hull_polygon


def project_point_onto_line(point, line):
    """Project a point onto a line segment and return the projected point."""
    line_start, line_end = np.array(line.coords[0]), np.array(line.coords[1])
    line_vec = line_end - line_start
    point_vec = np.array(point.coords[0]) - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)    
    t = np.clip(t, 0.0, 1.0)
    nearest = line_start + t * line_vec
    return Point(nearest)


def find_nearest_projection_on_polyline(point, polyline):
    """Find the nearest projected point of a point onto a polyline."""
    min_dist = float('inf')
    nearest_point = None
    for i in range(len(polyline.coords) - 1):
        segment = LineString(polyline.coords[i:i+2])
        proj_point = project_point_onto_line(point, segment)
        dist = point.distance(proj_point)
        if dist < min_dist:
            min_dist = dist
            nearest_point = proj_point
    return np.array(nearest_point.coords)


def find_and_sort_intersections(segmenet1, segment2):
    # Convert polylines to LineString objects

    # Find the intersection between the two LineStrings
    intersection = segmenet1.intersection(segment2)

    # Prepare a list to store intersection points
    intersections = []

    # Check the type of intersection
    if "Point" in intersection.geom_type:
        # Single point or multiple points
        if intersection.geom_type == "MultiPoint":
            intersections.extend(list(intersection))
        else:
            intersections.append(intersection)
    elif "LineString" in intersection.geom_type:
        # In case of lines or multiline, get boundary points (start and end points of line segments)
        if intersection.geom_type == "MultiLineString":
            for line in intersection:
                intersections.extend(list(line.boundary))
        else:
            intersections.extend(list(intersection.boundary))

    # Remove duplicates and ensure they are Point objects
    unique_intersections = [Point(coords) for coords in set(pt.coords[0] for pt in intersections)]

    # Sort the intersection points by their distance along the first polyline
    sorted_intersections = sorted(unique_intersections, key=lambda pt: segmenet1.project(pt))

    return sorted_intersections


def get_intersection_point_on_line(line, intersection):
    intersection_points  = find_and_sort_intersections(LineString(line), intersection)
    if len(intersection_points) >= 2:
        line_intersect_start = intersection_points[0]
        line_intersect_end = intersection_points[-1]
    elif len(intersection_points) == 1:
        if intersection.contains(Point(line[0])):
            line_intersect_start = Point(line[0])
            line_intersect_end = intersection_points[0]
        elif intersection.contains(Point(line[-1])):
            line_intersect_start = Point(line[-1])
            line_intersect_end = intersection_points[0]
        else:
            return None, None            
    else:
        return None, None            
    return line_intersect_start, line_intersect_end

def merge_l2_points_to_l1(line1, line2, line2_intersect_start, line2_intersect_end):
    # get nearest point on line2 to line2_intersect_start
    line2_point_to_merge = []
    line2_intersect_start_dis = line2.project(line2_intersect_start)
    line2_intersect_end_dis = line2.project(line2_intersect_end)
    for point in np.array(line2.coords):
        point_geom = Point(point)
        dis = line2.project(point_geom)
        if dis > line2_intersect_start_dis and dis < line2_intersect_end_dis:
            line2_point_to_merge.append(point)
            
    # merged the points
    merged_line2_points = []
    for point in line2_point_to_merge:
        # Use the `project` method to find the distance along the polyline to the closest point
        point_geom = Point(point)
        # Use the `interpolate` method to find the actual point on the polyline
        closest_point_on_line = find_nearest_projection_on_polyline(point_geom, line1)
        if len(closest_point_on_line) == 0:
            merged_line2_points.append(point)
        else:
            merged_line2_points.append(((closest_point_on_line + point) / 2)[0])

    if len(merged_line2_points) == 0:
        merged_line2_points = np.array([]).reshape(0, 2)
    else:
        merged_line2_points = np.array(merged_line2_points)
        
    return merged_line2_points        

def segment_line_based_on_merged_area(line, merged_points):
    
    if len(merged_points) == 0:
        return  np.array(line.coords),  np.array([]).reshape(0, 2)
    
    first_merged_point = merged_points[0]
    last_merged_point = merged_points[-1]
    
    start_dis = line.project(Point(first_merged_point))
    end_dis = line.project(Point(last_merged_point))
    
    start_segmenet = []
    for point in np.array(line.coords):
        point_geom = Point(point)
        if line.project(point_geom) < start_dis:
            start_segmenet.append(point)
    
    end_segmenet = []
    for point in np.array(line.coords):
        point_geom = Point(point)
        if line.project(point_geom) > end_dis:
            end_segmenet.append(point)
            
    if len(start_segmenet) == 0:
        start_segmenet = np.array([]).reshape(0, 2)
    else:
        start_segmenet = np.array(start_segmenet)
        
    if len(end_segmenet) == 0:
        end_segmenet = np.array([]).reshape(0, 2)
    else:
        end_segmenet = np.array(end_segmenet)
    
    return start_segmenet, end_segmenet
    
def get_bbox_size_for_points(points):
    if len(points) == 0:
        return 0, 0
    
    # Initialize min and max coordinates with the first point
    min_x, min_y = points[0]
    max_x, max_y = points[0]

    # Iterate through each point to update min and max coordinates
    for x, y in points[1:]:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    return max_x - min_x, max_y - min_y

def get_longer_segmenent_to_merged_points(l1_segment, l2_segment, merged_line2_points, segment_type="start"):
    # remove points from segments if it's too close to merged_line2_points
    l1_segment_temp = []
    if len(merged_line2_points) > 1:
        merged_polyline = LineString(merged_line2_points)
        for point in l1_segment:
            if merged_polyline.distance(Point(point)) > 0.1:
                l1_segment_temp.append(point)
    elif len(merged_line2_points) == 1:
        for point in l1_segment:
            if Point(point).distance(Point(merged_line2_points[0])) > 0.1:
                l1_segment_temp.append(point)
    elif len(merged_line2_points) == 0:
        l1_segment_temp = l1_segment
        
                
    l1_segment = np.array(l1_segment_temp)
    
    l2_segmenet_temp = []
    if len(merged_line2_points) > 1:
        merged_polyline = LineString(merged_line2_points)
        for point in l2_segment:
            if merged_polyline.distance(Point(point)) > 0.1:
                l2_segmenet_temp.append(point)
    elif len(merged_line2_points) == 1:
        for point in l2_segment:
            if Point(point).distance(Point(merged_line2_points[0])) > 0.1:
                l2_segmenet_temp.append(point)
    elif len(merged_line2_points) == 0:
        l2_segmenet_temp = l2_segment
                
    l2_segment = np.array(l2_segmenet_temp)
    
    if segment_type == "start":
        
        temp = l1_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[0])
        
        l1_start_box_size = get_bbox_size_for_points(temp)
        
        temp = l2_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[0])
        l2_start_box_size = get_bbox_size_for_points(temp)
    
        if l2_start_box_size[0]*l2_start_box_size[1] >= l1_start_box_size[0]*l1_start_box_size[1]:
            longer_segment = l2_segment
        else:
            longer_segment = l1_segment
    else:
        temp = l1_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[-1])
        l1_end_box_size = get_bbox_size_for_points(temp)
        
        temp = l2_segment.tolist()
        if len(merged_line2_points) > 0:
            temp.append(merged_line2_points[-1])
        l2_end_box_size = get_bbox_size_for_points(temp)
    
        if l2_end_box_size[0]*l2_end_box_size[1] >= l1_end_box_size[0]*l1_end_box_size[1]:
            longer_segment = l2_segment
        else:
            longer_segment = l1_segment
    
    if len(longer_segment) == 0:
        longer_segment = np.array([]).reshape(0, 2)
    else:
        longer_segment = np.array(longer_segment)
        
    return longer_segment
    
def get_line_lineList_max_intersection(merged_lines, line, thickness=4):
    pre_line = merged_lines[-1]
    max_iou = 0
    merged_line_index = 0
    for line_index, one_merged_line in enumerate(merged_lines):
        line1 = LineString(one_merged_line)
        line2 = LineString(line)
        thick_line1 = line1.buffer(thickness)
        thick_line2 = line2.buffer(thickness)
        intersection = thick_line1.intersection(thick_line2)
        if intersection.area / thick_line2.area > max_iou:
            max_iou = intersection.area / thick_line2.area
            pre_line = np.array(line1.coords)
            merged_line_index = line_index
    return intersection, pre_line, merged_line_index
    
def algin_l2_with_l1(line1, line2):
    
    if len(line1) > len(line2):
        l2_len = len(line2)
        line1_geom = LineString(line1)
        interval_length = line1_geom.length / (l2_len - 1)
        line1 = [np.array(line1_geom.interpolate(interval_length * i)) for i in range(l2_len)]
        
    elif len(line1) < len(line2):
        l1_len = len(line1)
        line2_geom = LineString(line2)
        interval_length = line2_geom.length / (l1_len - 1)
        line2 = [np.array(line2_geom.interpolate(interval_length * i)) for i in range(l1_len)]
    
    # make line1 and line2 same direction, pre_line.coords[0] shold be closer to line2.coords[0]
    line1_geom = LineString(line1)
    line2_flip = np.flip(line2, axis=0)
    
    line2_traj_len = 0
    for point_idx, point in enumerate(line2):
        line2_traj_len += np.linalg.norm(point - line1[point_idx])
    
    flip_line2_traj_len = 0
    for point_idx, point in enumerate(line2_flip):
        flip_line2_traj_len += np.linalg.norm(point - line1[point_idx])
    
        
    if abs(flip_line2_traj_len - line2_traj_len) < 3:
        # get the trajectory length
        line2_walk_len = 0
        for point in line2:
            point_geom = Point(point)
            proj_point = find_nearest_projection_on_polyline(point_geom, line1_geom)
            if len(proj_point) != 0:
                line2_walk_len += line1_geom.project(Point(proj_point[0]))
        
        flip_line2_walk_len = 0
        for point in line2:
            point_geom = Point(point)
            proj_point = find_nearest_projection_on_polyline(point_geom, line1_geom)
            if len(proj_point) != 0:
                flip_line2_walk_len += line1_geom.project(Point(proj_point[0]))
        
        if flip_line2_walk_len < line2_walk_len:
            return line2_flip
        else:
            return line2
        
    
    if flip_line2_traj_len < line2_traj_len:
        return line2_flip
    else:
        return line2

def _is_u_shape(line, direction):
    assert direction in ['left', 'right'], 'Wrong direction argument {}'.format(direction)
    line_geom = LineString(line)
    length = line_geom.length
    mid_point = np.array(line_geom.interpolate(length / 2).coords)[0]
    start = line[0]
    end = line[-1]

    if direction == 'left':
        cond1 = mid_point[0] < start[0] and mid_point[0] < end[0]
    else:
        cond1 = mid_point[0] > start[0] and mid_point[0] > end[0]
    
    dist_start_end = np.sqrt((start[0] - end[0])**2 + (start[1]-end[1])**2)
    cond2 = length >= math.pi / 2 * dist_start_end

    return cond1 and cond2

def check_circle(pre_line, vec):

    # if the last line in merged_lines is a circle
    if np.linalg.norm(pre_line[0] - pre_line[-1]) == 0:
        return True
    
    # if the last line in merged_lines is almost a circle and the new line is close to the circle
    if np.linalg.norm(pre_line[0] - pre_line[-1]) < 0.1:
        vec_2_circle_distance = 0
        for point in vec:
            vec_2_circle_distance += LineString(pre_line).distance(Point(point))
        if vec_2_circle_distance < 3:
            return True
    return False
        
def connect_polygon(merged_polyline, merged_lines):
    start_end_connect = [merged_polyline[0], merged_polyline[-1]]
    iou = []
    length_ratio = []
    for one_merged_line in merged_lines:
        line1 = LineString(one_merged_line)
        line2 = LineString(start_end_connect)
        thickness = 1
        thick_line1 = line1.buffer(thickness)
        thick_line2 = line2.buffer(thickness)
        intersection = thick_line1.intersection(thick_line2)
        iou.append(intersection.area / thick_line2.area)
        length_ratio.append(line1.length / line2.length)

    if max(iou) > 0.95 and max(length_ratio) > 3.0:
        merged_polyline = np.concatenate((merged_polyline, [merged_polyline[0]]), axis=0)
    return merged_polyline
    
def iou_merge_boundry(merged_lines, vec, thickness=1):

    # intersection : the intersection area between the new line and the line in the merged_lines; is a polygon
    intersection, pre_line, merged_line_index = get_line_lineList_max_intersection(merged_lines, vec, thickness)

    # corner case: check if the last line in merged_lines is a circle
    if check_circle(pre_line, vec):
        return merged_lines

    # Handle U-shape, the main corner case
    if _is_u_shape(pre_line, 'left'):
        if _is_u_shape(vec, 'right'):
            # Two u shapes with opposite directions, directly generate a polygon exterior
            polygon = find_largest_convex_hull([LineString(pre_line), LineString(vec)])
            merged_lines[-1] = np.array(polygon.exterior.coords)
            return merged_lines
        elif not _is_u_shape(vec, 'left'):
            line_geom1 = LineString(pre_line)
            line1_dists = np.array([line_geom1.project(Point(x)) for x in pre_line])
            split_mask = line1_dists > line_geom1.length / 2
            split_1 = LineString(pre_line[~split_mask])
            split_2 = LineString(pre_line[split_mask])

            # get the projected distance
            np1 = np.array(nearest_points(split_1, Point(Point(pre_line[-1])))[0].coords)[0]
            np2 = np.array(nearest_points(split_2, Point(Point(pre_line[0])))[0].coords)[0]
            dist1 = np.linalg.norm(np1-pre_line[-1])
            dist2 = np.linalg.norm(np2-pre_line[0])
            dist = min(dist1, dist2)

            if dist < thickness:
                line_geom2 = LineString(vec)
                dist1 = line_geom2.distance(Point(pre_line[0]))
                dist2 = line_geom2.distance(Point(pre_line[-1]))
                pt = pre_line[0] if dist1 <= dist2 else pre_line[-1]
                if vec[0][0] > vec[1][0]:
                    vec = np.array(vec[::-1])
                    line_geom2 = LineString(vec)
                proj_length = line_geom2.project(Point(pt))
                l2_select_mask = np.array([line_geom2.project(Point(x)) > proj_length for x in vec])
                selected_l2 = vec[l2_select_mask]
                merged_result = np.concatenate([pre_line[:-1, :], pt[None, ...], selected_l2], axis=0)
                merged_lines[-1] = merged_result
                return merged_lines
    
    # align the new line with the line in the merged_lines so that points on two lines are traversed in the same direction
    vec = algin_l2_with_l1(pre_line, vec)
    line1 = LineString(pre_line)
    line2 = LineString(vec)
    
    # get the intersection points between IOU area and two lines
    line1_intersect_start, line1_intersect_end = get_intersection_point_on_line(pre_line, intersection)
    line2_intersect_start, line2_intersect_end = get_intersection_point_on_line(vec, intersection)
    
    # If no intersection points are found, use the last point of the line1 and the first point of the line2 as the intersection points --> this is a corner case that we will connect the two lines head to tail directly
    if line1_intersect_start is None or line1_intersect_end is None or line2_intersect_start is None or line2_intersect_end is None:
        line1_intersect_start = Point(pre_line[-1])
        line1_intersect_end = Point(pre_line[-1])
        line2_intersect_start = Point(vec[0])
        line2_intersect_end = Point(vec[0])
    
    # merge the points on line2's intersection area towards line1
    merged_line2_points = merge_l2_points_to_l1(line1, line2, line2_intersect_start, line2_intersect_end)
    # merge the points on line1's intersection area towards line2
    merged_line1_points = merge_l2_points_to_l1(line2, line1, line1_intersect_start, line1_intersect_end)
    
    # segment the lines based on the merged points (intersection area); split the line in to start segment and merged segment and end segment
    l2_start_segment, l2_end_segment = segment_line_based_on_merged_area(line2, merged_line2_points)
    l1_start_segment, l1_end_segment = segment_line_based_on_merged_area(line1, merged_line1_points)
    
    # choose the longer segment between line1 and line2 to be the final start segment and end segment
    start_segment = get_longer_segmenent_to_merged_points(l1_start_segment, l2_start_segment, merged_line2_points, segment_type="start")
    end_segment = get_longer_segmenent_to_merged_points(l1_end_segment, l2_end_segment, merged_line2_points, segment_type="end")
    merged_polyline = np.concatenate((start_segment, merged_line2_points, end_segment), axis=0)
    
    # corner case : check if need to connect the polyline to form a circle
    merged_polyline = connect_polygon(merged_polyline, merged_lines)
    
    merged_lines[merged_line_index] = merged_polyline
  
    return merged_lines

def iou_merge_divider(merged_lines, vec, thickness=1):
    # intersection : the intersection area between the new line and the line in the merged_lines; is a polygon
    # pre_line : the line in merged_lines that has max IOU with the new line
    intersection, pre_line, merged_line_index = get_line_lineList_max_intersection(merged_lines, vec, thickness)
    # align the new line with the line in the merged_lines so that points on two lines are traversed in the same direction
    vec = algin_l2_with_l1(pre_line, vec)
    
    line1 = LineString(pre_line)
    line2 = LineString(vec)
    
    # get the intersection points between IOU area and two lines
    line1_intersect_start, line1_intersect_end = get_intersection_point_on_line(pre_line, intersection)
    line2_intersect_start, line2_intersect_end = get_intersection_point_on_line(vec, intersection)
    
    # If no intersection points are found, use the last point of the line1 and the first point of the line2 as the intersection points --> this is a corner case that we will connect the two lines head to tail directly
    if line1_intersect_start is None or line1_intersect_end is None or line2_intersect_start is None or line2_intersect_end is None:
        line1_intersect_start = Point(pre_line[-1])
        line1_intersect_end = Point(pre_line[-1])
        line2_intersect_start = Point(vec[0])
        line2_intersect_end = Point(vec[0])
    
    # merge the points on line2's intersection area towards line1
    merged_line2_points = merge_l2_points_to_l1(line1, line2, line2_intersect_start, line2_intersect_end)
    # merge the points on line1's intersection area towards line2
    merged_line1_points = merge_l2_points_to_l1(line2, line1, line1_intersect_start, line1_intersect_end)
    
    # segment the lines based on the merged points (intersection area); split the line in to start segment and merged segment and end segment
    l2_start_segment, l2_end_segment = segment_line_based_on_merged_area(line2, merged_line2_points)
    l1_start_segment, l1_end_segment = segment_line_based_on_merged_area(line1, merged_line1_points)
    
    # choose the longer segment between line1 and line2 to be the final start segment and end segment
    start_segment = get_longer_segmenent_to_merged_points(l1_start_segment, l2_start_segment, merged_line2_points, segment_type="start")
    end_segment = get_longer_segmenent_to_merged_points(l1_end_segment, l2_end_segment, merged_line2_points, segment_type="end")
    merged_polyline = np.concatenate((start_segment, merged_line2_points, end_segment), axis=0)
    
    # update the merged_lines
    merged_lines[merged_line_index] = merged_polyline
    
    return merged_lines

def merge_divider(vecs=None, thickness=1):
    merged_lines = []
    for vec in vecs:
        
        # if the merged_lines is empty, add the first line
        if len(merged_lines) == 0:
            merged_lines.append(vec)
            continue
        
        # thicken the vec (the new line) and the merged_lines calculate the max IOU between the new line and the merged_lines
        iou = []
        for one_merged_line in merged_lines:
            line1 = LineString(one_merged_line)
            line2 = LineString(vec)
            thick_line1 = line1.buffer(thickness)
            thick_line2 = line2.buffer(thickness)
            intersection = thick_line1.intersection(thick_line2)
            iou.append(intersection.area / thick_line2.area)
        
        # If the max IOU is 0, add the new line to the merged_lines
        if max(iou) == 0:
            merged_lines.append(vec)
        # If IOU is not 0, merge the new line with the line in the merged_lines
        else:
            merged_lines = iou_merge_divider(merged_lines, vec, thickness=thickness)

           
    return merged_lines

def merge_boundary(vecs=None, thickness=1, iou_threshold=0.95):
    merged_lines = []
    for vec in vecs:

        # if the merged_lines is empty, add the first line
        if len(merged_lines) == 0:
            merged_lines.append(vec)
            continue
        
        # thicken the vec (the new line) and the merged_lines calculate the max IOU between the new line and the merged_lines
        iou = []
        for one_merged_line in merged_lines:
            line1 = LineString(one_merged_line)
            line2 = LineString(vec)
            thick_line1 = line1.buffer(thickness)
            thick_line2 = line2.buffer(thickness)
            intersection = thick_line1.intersection(thick_line2)
            iou.append(intersection.area / thick_line2.area)
        
        # If the max IOU larger than the threshold, skip the new line
        if max(iou) > iou_threshold:
            continue
        
        # If IOU is not 0, merge the new line with the line in the merged_lines
        if max(iou) > 0:
            merged_lines = iou_merge_boundry(merged_lines, vec, thickness=thickness)
        else:
            merged_lines.append(vec)
           
    return merged_lines

def get_consecutive_vectors_with_opt(prev_vectors=None,prev2curr_matrix=None,origin=None,roi_size=None, denormalize=False, clip=False):
    # transform prev vectors
    prev2curr_vectors = dict()
    for label, vecs in prev_vectors.items():
        if len(vecs) > 0:
            vecs = np.stack(vecs, 0)
            vecs = torch.tensor(vecs)
            N, num_points, _ = vecs.shape
            if denormalize:
                denormed_vecs = vecs * roi_size + origin # (num_prop, num_pts, 2)
            else:
                denormed_vecs = vecs
            denormed_vecs = torch.cat([
                denormed_vecs,
                denormed_vecs.new_zeros((N, num_points, 1)), # z-axis
                denormed_vecs.new_ones((N, num_points, 1)) # 4-th dim
            ], dim=-1) # (num_prop, num_pts, 4)

            transformed_vecs = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_vecs.double()).float()
            normed_vecs = (transformed_vecs[..., :2] - origin) / roi_size # (num_prop, num_pts, 2)
            if clip:
                normed_vecs = torch.clip(normed_vecs, min=0., max=1.)
            prev2curr_vectors[label] = normed_vecs
        else:
            prev2curr_vectors[label] = vecs

    # convert to ego space for visualization
    for label in prev2curr_vectors:
        if len(prev2curr_vectors[label]) > 0:
            prev2curr_vectors[label] = prev2curr_vectors[label] * roi_size + origin
    return prev2curr_vectors

def get_prev2curr_vectors(vecs=None, prev2curr_matrix=None,origin=None,roi_size=None, denormalize=False, clip=False):
    # transform prev vectors
    if len(vecs) > 0:
        vecs = np.stack(vecs, 0)
        vecs = torch.tensor(vecs)
        N, num_points, _ = vecs.shape
        if denormalize:
            denormed_vecs = vecs * roi_size + origin # (num_prop, num_pts, 2)
        else:
            denormed_vecs = vecs
        denormed_vecs = torch.cat([
            denormed_vecs,
            denormed_vecs.new_zeros((N, num_points, 1)), # z-axis
            denormed_vecs.new_ones((N, num_points, 1)) # 4-th dim
        ], dim=-1) # (num_prop, num_pts, 4)

        transformed_vecs = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_vecs.double()).float()
        vecs = (transformed_vecs[..., :2] - origin) / roi_size # (num_prop, num_pts, 2)
        if clip:
            vecs = torch.clip(vecs, min=0., max=1.)
        # vecs = vecs * roi_size + origin
    
    return vecs

def plot_fig_merged_per_frame(num_frames, car_trajectory, x_min, x_max, y_min, y_max, pred_save_folder, id_prev2curr_pred_vectors, id_prev2curr_pred_frame, args):
    os.makedirs(pred_save_folder, exist_ok=True)
  
    # key the current status of the instance, add into the dict when it first appears
    instance_bank = dict()

    # trace the path reversely, get the sub-sampled traj for visualizing the car
    pre_center = car_trajectory[-1][0]
    selected_traj_timesteps = []
    for timestep, (car_center, rotation_degrees) in enumerate(car_trajectory[::-1]):
        if np.linalg.norm(car_center - pre_center) < 5 and timestep > 0 and timestep < len(car_trajectory)-1:
            continue
        selected_traj_timesteps.append(len(car_trajectory)-1-timestep)
        pre_center = car_center
    selected_traj_timesteps = selected_traj_timesteps[::-1]

    image_list = [pred_save_folder + f'/{frame_timestep}.png' for frame_timestep in range(num_frames)]
    #save_t(len(image_list), pred_save_folder) # save the timestep text mp4 file

    # plot the figure at each frame
    for frame_timestep in range(num_frames):
        plt.figure(facecolor='lightgreen')
        fig = plt.figure(figsize=(int(abs(x_min) + abs(x_max)) + 10 , int(abs(y_min) + abs(y_max)) + 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # setup the figure with car
        car_img = Image.open('resources/car-orange.png')
        faded_rate = np.linspace(0.2, 1, num=len(car_trajectory))
        pre_center = car_trajectory[0][0]

        for t in selected_traj_timesteps: # only plot the car at the selected timesteps
            if t > frame_timestep: # if the car has not appeared at this frame
                break
            car_center, rotation_degrees = car_trajectory[t]
            translation = transforms.Affine2D().translate(car_center[0], car_center[1])
            rotation = transforms.Affine2D().rotate_deg(rotation_degrees)
            rotation_translation = rotation + translation
            ax.imshow(car_img, extent=[-2.2, 2.2, -2, 2], transform=rotation_translation+ ax.transData, alpha=faded_rate[t])
        
        for vec_tag, vec_all_frames in id_prev2curr_pred_vectors.items():
            vec_frame_info = id_prev2curr_pred_frame[vec_tag] 
            first_appear_frame = sorted(list(vec_frame_info.keys()))[0]

            need_merge = False
            if frame_timestep < first_appear_frame : # the instance has not appeared
                continue
            elif frame_timestep in vec_frame_info:
                need_merge = True
                vec_index_in_instance = vec_frame_info[frame_timestep]

            label, vec_glb_idx = vec_tag.split('_')
            label = int(label)
            vec_glb_idx = int(vec_glb_idx)

            if need_merge:
                curr_vec = vec_all_frames[vec_index_in_instance]
                curr_vec_polyline = LineString(curr_vec)
                if vec_tag not in instance_bank: # if the instance first appears
                    polylines = [curr_vec_polyline,]
                else: # if the instance has appeared before, polylines = previous merged polyline + current polyline
                    polylines = instance_bank[vec_tag] + [curr_vec_polyline,]
            else: # if the instance has not appeared in this frame
                polylines = instance_bank[vec_tag]

            if label == 0: # ped_crossing
                color = 'b'
            elif label == 1: # divider
                color = 'r'
            elif label == 2: # boundary
                color = 'g'
            
            if label == 0: # crossing, merged by convex hull
                if need_merge:
                    polygon = merge_corssing(polylines)
                    polygon = polygon.simplify(args.simplify)
                    vector = np.array(polygon.exterior.coords) 
                else: # if no new instance, use the previous merged polyline to plot
                    vector = np.array(polylines[0].coords) 

                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=args.line_opacity)
                ax.plot(x, y, "o", color=color, markersize=50)

                # update instance bank for ped
                updated_polyline = LineString(vector)
                instance_bank[vec_tag] = [updated_polyline, ]

            elif label == 1: # divider, merged fitting a polyline
                if need_merge:
                    polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
                    polylines_vecs = merge_divider(polylines_vecs)
                else:  # if no new instance, use the previous merged polyline to plot
                    polylines_vecs = [np.array(line.coords) for line in polylines]

                for one_line in polylines_vecs:
                    one_line = np.array(LineString(one_line).simplify(args.simplify*2).coords)
                    pts = one_line[:, :2]
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=args.line_opacity)
                    ax.plot(x, y, "o", color=color, markersize=50)

                # update instance bank for line
                updated_polylines = [LineString(vec) for vec in polylines_vecs]
                instance_bank[vec_tag] = updated_polylines

            elif label == 2: # boundary, do not merge
                if need_merge:
                    polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
                    polylines_vecs = merge_boundary(polylines_vecs)
                else: # if no new instance, use the previous merged polyline to plot
                    polylines_vecs = [np.array(line.coords) for line in polylines]

                for one_line in polylines_vecs:
                    one_line = np.array(LineString(one_line).simplify(args.simplify).coords)
                    pts = one_line[:, :2]
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=args.line_opacity)
                    ax.plot(x, y, "o", color=color, markersize=50)

                # update instance bank for line
                updated_polylines = [LineString(vec) for vec in polylines_vecs]
                instance_bank[vec_tag] = updated_polylines
        
        pred_save_path = pred_save_folder + f'/{frame_timestep}.png'
        plt.grid(False)
        plt.savefig(pred_save_path, bbox_inches='tight', transparent=args.transparent, dpi=args.dpi)
        plt.clf() 
        plt.close(fig)
        print("image saved to : ", pred_save_path)

    image_list = [pred_save_folder + f'/{frame_timestep}.png' for frame_timestep in range(num_frames)]
    gif_output_path = pred_save_folder + '/vis.gif'
    save_as_video(image_list, gif_output_path)

# merge the vectors across all frames and plot the merged vectors
def plot_fig_merged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, id_prev2curr_pred_vectors, args):
                
    # setup the figure with car
    fig = plt.figure(figsize=(int(abs(x_min) + abs(x_max)) + 10 , int(abs(y_min) + abs(y_max)) + 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    car_img = Image.open('resources/car-orange.png')
    
    faded_rate = np.linspace(0.2, 1, num=len(car_trajectory))

    # trace the path reversely, get the sub-sampled traj for visualizing the car
    pre_center = car_trajectory[-1][0]
    selected_traj = []
    selected_timesteps = []
    for timestep, (car_center, rotation_degrees) in enumerate(car_trajectory[::-1]):
        if np.linalg.norm(car_center - pre_center) < 5 and timestep > 0 and timestep < len(car_trajectory)-1:
            continue
        selected_traj.append([car_center, rotation_degrees])
        selected_timesteps.append(len(car_trajectory)-1-timestep)
        pre_center = car_center
    selected_traj = selected_traj[::-1]
    selected_timesteps = selected_timesteps[::-1]

    for selected_t, (car_center, rotation_degrees) in zip(selected_timesteps, selected_traj):
        translation = transforms.Affine2D().translate(car_center[0], car_center[1])
        rotation = transforms.Affine2D().rotate_deg(rotation_degrees)
        rotation_translation = rotation + translation
        ax.imshow(car_img, extent=[-2.2, 2.2, -2, 2], transform=rotation_translation+ ax.transData, 
                alpha=faded_rate[selected_t])
    
    # merge the vectors across all frames
    for tag, vecs in id_prev2curr_pred_vectors.items():
        label, vec_glb_idx = tag.split('_')
        label = int(label)
        vec_glb_idx = int(vec_glb_idx)
        

        if label == 0: # ped_crossing
            color = 'b'
        elif label == 1: # divider
            color = 'r'
        elif label == 2: # boundary
            color = 'g'
    
        # get the vectors belongs to the same instance
        polylines = []
        for vec in vecs:
            polylines.append(LineString(vec))
        if len(polylines) <= 0:
            continue

        if label == 0: # crossing, merged by convex hull
            polygon = merge_corssing(polylines)
            if polygon.area < 2:
                continue
            polygon = polygon.simplify(args.simplify)
            vector = np.array(polygon.exterior.coords) 
            pts = vector[:, :2]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=args.line_opacity)
            ax.plot(x, y, "o", color=color, markersize=50)
        elif label == 1: # divider, merged by interpolation
            polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
            polylines_vecs = merge_divider(polylines_vecs)
            for one_line in polylines_vecs:
                one_line = np.array(LineString(one_line).simplify(args.simplify).coords)
                pts = one_line[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=args.line_opacity)
                ax.plot(x, y, "o", color=color, markersize=50)
        elif label == 2: # boundary, merged by interpolation
            polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
            polylines_vecs = merge_boundary(polylines_vecs)
            for one_line in polylines_vecs:
                one_line = np.array(LineString(one_line).simplify(args.simplify).coords)
                pts = one_line[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=args.line_opacity)
                ax.plot(x, y, "o", color=color, markersize=50)

    plt.grid(False)
    plt.savefig(pred_save_path, bbox_inches='tight', transparent=args.transparent, dpi=args.dpi)
    plt.clf() 
    plt.close(fig)
    print("image saved to : ", pred_save_path)

def plot_fig_unmerged_per_frame(num_frames, car_trajectory, x_min, x_max, y_min, y_max, pred_save_folder, id_prev2curr_pred_vectors, id_prev2curr_pred_frame, args):

    os.makedirs(pred_save_folder, exist_ok=True)

    # trace the path reversely, get the sub-sampled traj for visualizing the car
    pre_center = car_trajectory[-1][0]
    selected_traj_timesteps = []
    for timestep, (car_center, rotation_degrees) in enumerate(car_trajectory[::-1]):
        if np.linalg.norm(car_center - pre_center) < 5 and timestep > 0 and timestep < len(car_trajectory)-1:
            continue
        selected_traj_timesteps.append(len(car_trajectory)-1-timestep)
        pre_center = car_center
    selected_traj_timesteps = selected_traj_timesteps[::-1]

    # setup the figure with car
    fig = plt.figure(figsize=(int(abs(x_min) + abs(x_max)) + 10 , int(abs(y_min) + abs(y_max)) + 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    car_img = Image.open('resources/car-orange.png')


    for frame_timestep in range(num_frames):

        faded_rate = np.linspace(0.2, 1, num=len(car_trajectory))
        if frame_timestep in selected_traj_timesteps:
            car_center, rotation_degrees = car_trajectory[frame_timestep]
            translation = transforms.Affine2D().translate(car_center[0], car_center[1])
            rotation = transforms.Affine2D().rotate_deg(rotation_degrees)
            rotation_translation = rotation + translation
            ax.imshow(car_img, extent=[-2.2, 2.2, -2, 2], transform=rotation_translation+ ax.transData, alpha=faded_rate[frame_timestep])
        
        # plot the vectors
        for vec_tag, vec_all_frames in id_prev2curr_pred_vectors.items():
            vec_frame_info = id_prev2curr_pred_frame[vec_tag]
            if frame_timestep not in vec_frame_info: # the instance has not appeared
                continue
            else:
                vec_index_in_instance = vec_frame_info[frame_timestep]
            
            curr_vec = vec_all_frames[vec_index_in_instance]
            label, vec_glb_idx = vec_tag.split('_')
            label = int(label)
            vec_glb_idx = int(vec_glb_idx)
            
            if label == 0: # ped_crossing
                color = 'b'
            elif label == 1: # divider
                color = 'r'
            elif label == 2: # boundary
                color = 'g'
            
            polyline = LineString(curr_vec)
            vector = np.array(polyline.coords)
            pts = vector[:, :2]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            ax.plot(x, y, 'o-', color=color, linewidth=20, markersize=50)

        pred_save_path = pred_save_folder + f'/{frame_timestep}.png'
        plt.savefig(pred_save_path, bbox_inches='tight', transparent=args.transparent, dpi=args.dpi)
        print("image saved to : ", pred_save_path)

    plt.grid(False)
    plt.clf() 
    plt.close(fig)
    image_list = [pred_save_folder + f'/{frame_timestep}.png' for frame_timestep in range(num_frames)]
    gif_output_path = pred_save_folder + '/vis.gif'
    save_as_video(image_list, gif_output_path)


def plot_fig_unmerged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, id_prev2curr_pred_vectors, args):
                
    # setup the figure with car
    fig = plt.figure(figsize=(int(abs(x_min) + abs(x_max)) + 10 , int(abs(y_min) + abs(y_max)) + 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    car_img = Image.open('resources/car-orange.png')
    
    # trace the path reversely, get the sub-sampled traj for visualizing the car 
    pre_center = car_trajectory[-1][0]
    selected_traj = []
    selected_timesteps = []
    for timestep, (car_center, rotation_degrees) in enumerate(car_trajectory[::-1]):
        if np.linalg.norm(car_center - pre_center) < 5 and timestep > 0 and timestep < len(car_trajectory)-1:
            continue
        selected_traj.append([car_center, rotation_degrees])
        selected_timesteps.append(len(car_trajectory)-1-timestep)
        pre_center = car_center
    selected_traj = selected_traj[::-1]
    selected_timesteps = selected_timesteps[::-1]

    # plot the car trajectory with faded_rate 
    faded_rate = np.linspace(0.2, 1, num=len(car_trajectory))
    for selected_t, (car_center, rotation_degrees) in zip(selected_timesteps, selected_traj):
        translation = transforms.Affine2D().translate(car_center[0], car_center[1])
        rotation = transforms.Affine2D().rotate_deg(rotation_degrees)
        rotation_translation = rotation + translation
        ax.imshow(car_img, extent=[-2.2, 2.2, -2, 2], transform=rotation_translation+ ax.transData, 
                alpha=faded_rate[selected_t])
    
    # plot the unmerged vectors (all the predicted/ gt vectors)
    for tag, vecs in id_prev2curr_pred_vectors.items():
        label, vec_glb_idx = tag.split('_')
        label = int(label)
        vec_glb_idx = int(vec_glb_idx)

        if label == 0: # ped_crossing
            color = 'b'
        elif label == 1: # divider
            color = 'r'
        elif label == 2: # boundary
            color = 'g'
        
        polylines = []
        for vec in vecs:
            polylines.append(LineString(vec))
            
        if len(polylines) <= 0:
            continue

        for one_line in polylines:
            vector = np.array(one_line.coords)
            pts = vector[:, :2]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            ax.plot(x, y, 'o-', color=color, linewidth=20, markersize=50)
        

    plt.savefig(pred_save_path, bbox_inches='tight', transparent=args.transparent, dpi=args.dpi)
    plt.clf()  
    plt.close(fig)
    print("image saved to : ", pred_save_path)

# the timestep text visualization
def save_t(t_max, main_save_folder):
    txt_save_folder = os.path.join(main_save_folder, 'txt')
    os.makedirs(txt_save_folder, exist_ok=True)
    t = range(t_max)

    for i in t:
        fig, ax = plt.subplots(figsize=(2, 1), dpi=300)  # Increase DPI for higher resolution
        ax.text(0.1, 0.5, f't = {i}', fontsize=40,ha='left', va='center')
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins around the text
        plt.savefig(f'{txt_save_folder}/text_{i}.png',pad_inches=0)
        plt.close(fig)

    text_images = [f'{txt_save_folder}/text_{i}.png' for i in t]
    frames = [imageio.imread(img_path) for img_path in text_images]
    mp4_output_path = os.path.join(main_save_folder, 'text.mp4')
    imageio.mimsave(mp4_output_path, frames, fps=10)  # fps controls the speed of the video
    print("mp4 saved to : ", mp4_output_path)

def save_as_video(image_list, mp4_output_path, scale=None):
    mp4_output_path = mp4_output_path.replace('.gif','.mp4')
    images = [Image.fromarray(imageio.imread(img_path)).convert("RGBA") for img_path in image_list]

    if scale is not None:
        w, h = images[0].size
        images = [img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS) for img in images]
    # images = [Image.new('RGBA', images[0].size, (255, 255, 255, 255))] + images
    
    try:
        imageio.mimsave(mp4_output_path, images,  format='MP4',fps=10)
    except ValueError: # in case the shapes are not the same, have to manually adjust
        resized_images = [img.resize(images[0].size, Image.Resampling.LANCZOS) for img in images]
        print('Size not all the same, manually adjust...')
        imageio.mimsave(mp4_output_path, resized_images,  format='MP4',fps=10)
    print("mp4 saved to : ", mp4_output_path)


def vis_pred_data(scene_name="", pred_results=None, origin=None, roi_size=None, args=None):
    

    # get the item index of the scene
    index_list = []
    for index in range(len(pred_results)):
        if pred_results[index]["scene_name"] == scene_name:
            index_list.append(index)
    
    car_trajectory = []
    id_prev2curr_pred_vectors = defaultdict(list)
    id_prev2curr_pred_frame_info = defaultdict(list)
    id_prev2curr_pred_frame = defaultdict(list)

    # iterate through each frame
    last_index = index_list[-1]
    for index in index_list:
        
        vectors = np.array(pred_results[index]["vectors"]).reshape((len(np.array(pred_results[index]["vectors"])), 20, 2))
        if abs(vectors.max()) <= 1:
            curr_vectors = vectors * roi_size + origin
        else:
            curr_vectors = vectors
            
        # get the transformation matrix of the last frame
        prev_e2g_trans =  torch.tensor(pred_results[index]['meta']['ego2global_translation'], dtype=torch.float64)
        prev_e2g_rot = torch.tensor(pred_results[index]['meta']['ego2global_rotation'], dtype=torch.float64)
        curr_e2g_trans  = torch.tensor(pred_results[last_index]['meta']['ego2global_translation'], dtype=torch.float64)
        curr_e2g_rot = torch.tensor(pred_results[last_index]['meta']['ego2global_rotation'], dtype=torch.float64)
        prev_e2g_matrix = torch.eye(4, dtype=torch.float64)
        prev_e2g_matrix[:3, :3] = prev_e2g_rot
        prev_e2g_matrix[:3, 3] = prev_e2g_trans

        curr_g2e_matrix = torch.eye(4, dtype=torch.float64)
        curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
        curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)
        
        prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
        prev2curr_pred_vectors = get_prev2curr_vectors(curr_vectors, prev2curr_matrix,origin,roi_size,False,False)
        prev2curr_pred_vectors = prev2curr_pred_vectors * roi_size + origin
        
        rotation_degrees = np.degrees(np.arctan2(prev2curr_matrix[:3, :3][1, 0], prev2curr_matrix[:3, :3][0, 0]))
        car_center = get_prev2curr_vectors(np.array((0,0)).reshape(1,1,2), prev2curr_matrix,origin,roi_size,False,False)* roi_size + origin
        car_trajectory.append([car_center.squeeze(), rotation_degrees])
        
        for i, (label, vec_glb_idx) in enumerate(zip(pred_results[index]['labels'], pred_results[index]['global_ids'])):
            dict_key = "{}_{}".format(label, vec_glb_idx)
            id_prev2curr_pred_vectors[dict_key].append(prev2curr_pred_vectors[i])
            id_prev2curr_pred_frame_info[dict_key].append([pred_results[index]["local_idx"], len(id_prev2curr_pred_frame[dict_key])])

        for key, frame_info in id_prev2curr_pred_frame_info.items():
            frame_localIdx = dict()
            for frame_time, local_index in frame_info:
                frame_localIdx[frame_time] = local_index
            id_prev2curr_pred_frame[key] = frame_localIdx
        
    
    # sort the id_prev2curr_pred_vectors
    id_prev2curr_pred_vectors = {key: id_prev2curr_pred_vectors[key] for key in sorted(id_prev2curr_pred_vectors)}

    
    # set the size of the image
    x_min = -roi_size[0] / 2
    x_max = roi_size[0] / 2
    y_min = -roi_size[1] / 2
    y_max = roi_size[1] / 2

    all_points = []
    for vecs in id_prev2curr_pred_vectors.values():
        points = np.concatenate(vecs, axis=0)
        all_points.append(points)
    all_points = np.concatenate(all_points, axis=0)

    x_min = min(x_min, all_points[:,0].min())
    x_max = max(x_max, all_points[:,0].max())
    y_min = min(y_min, all_points[:,1].min())
    y_max = max(y_max, all_points[:,1].max())

    scene_dir = os.path.join(args.out_dir, scene_name)
    os.makedirs(scene_dir,exist_ok=True)
    
    if args.per_frame_result:
        num_frames = len(index_list)
        pred_save_folder = os.path.join(scene_dir, f'pred_merged_per_frame')
        plot_fig_merged_per_frame(num_frames, car_trajectory, x_min, x_max, y_min, y_max, pred_save_folder, id_prev2curr_pred_vectors, id_prev2curr_pred_frame, args)
        pred_save_folder = os.path.join(scene_dir, f'pred_unmerged_per_frame')
        plot_fig_unmerged_per_frame(num_frames, car_trajectory, x_min, x_max, y_min, y_max, pred_save_folder, id_prev2curr_pred_vectors, id_prev2curr_pred_frame, args)
    pred_save_path = os.path.join(scene_dir, f'pred_unmerged.png')
    plot_fig_unmerged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, id_prev2curr_pred_vectors, args)
    pred_save_path = os.path.join(scene_dir, f'pred_merged.png')
    plot_fig_merged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, id_prev2curr_pred_vectors, args)
    comb_save_path = os.path.join(scene_dir, f'pred_comb.png')
    image_paths = [os.path.join(scene_dir, f'pred_merged.png'), os.path.join(scene_dir, f'pred_unmerged.png')]
    labels = ['Merged', 'Unmerged']
    combine_images_with_labels(image_paths, labels, comb_save_path)
    print("image saved to : ", comb_save_path)

def vis_gt_data(scene_name, args, dataset, gt_data, origin, roi_size):

    gt_info = gt_data[scene_name]
    gt_info_list = []
    ids_info = []

    # get the item index of the sample
    for index, one_idx in enumerate(gt_info["sample_ids"]):
        gt_info_list.append(dataset[one_idx])
        ids_info.append(gt_info["instance_ids"][index])

    car_trajectory = []
    scene_dir = os.path.join(args.out_dir,scene_name)
    os.makedirs(scene_dir,exist_ok=True)

    # key : label, vec_glb_idx ; value : list of vectors in the last frame's coordinate
    id_prev2curr_pred_vectors = defaultdict(list)
    # dict to store some information of the vectors
    id_prev2curr_pred_frame_info = defaultdict(list) 
    # key : label, vec_glb_idx ; value : {frame_time : idx of the vector; idx range from 0 to the number of vectors of the same instance }
    id_prev2curr_pred_frame = defaultdict(dict)

    scene_len = len(gt_info_list)
    for idx in range(scene_len):
        curr_vectors = dict()
        # denormalize the vectors
        for label, vecs in gt_info_list[idx]['vectors'].data.items():
            if len(vecs) > 0: # if vecs != []
                curr_vectors[label] = vecs * roi_size + origin
            else:
                curr_vectors[label] = vecs
        
        # get the transformation matrix of the last frame
        prev_e2g_trans = torch.tensor(gt_info_list[idx]['img_metas'].data['ego2global_translation'], dtype=torch.float64)
        prev_e2g_rot = torch.tensor(gt_info_list[idx]['img_metas'].data['ego2global_rotation'], dtype=torch.float64)
        curr_e2g_trans  = torch.tensor(gt_info_list[scene_len-1]['img_metas'].data['ego2global_translation'], dtype=torch.float64)
        curr_e2g_rot = torch.tensor(gt_info_list[scene_len-1]['img_metas'].data['ego2global_rotation'], dtype=torch.float64)
        prev_e2g_matrix = torch.eye(4, dtype=torch.float64)
        prev_e2g_matrix[:3, :3] = prev_e2g_rot
        prev_e2g_matrix[:3, 3] = prev_e2g_trans

        curr_g2e_matrix = torch.eye(4, dtype=torch.float64)
        curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
        curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)
        
        # get the transformed vectors from current frame to the last frame
        prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
        prev2curr_pred_vectors = get_consecutive_vectors_with_opt(curr_vectors,prev2curr_matrix,origin,roi_size,False,False)
        for label, id_info in ids_info[idx].items():
            for vec_local_idx, vec_glb_idx in id_info.items():
                dict_key = "{}_{}".format(label, vec_glb_idx)
                id_prev2curr_pred_vectors[dict_key].append(prev2curr_pred_vectors[label][vec_local_idx])
                # gt_info_list[idx]["seq_info"].data[1] stores the frame time that the vector appears
                id_prev2curr_pred_frame_info[dict_key].append([gt_info_list[idx]["seq_info"].data[1], len(id_prev2curr_pred_frame[dict_key])]) # set len(id_prev2curr_pred_frame[dict_key]) to be the index of the vector belongs to the same instance
        for key, frame_info in id_prev2curr_pred_frame_info.items():
            frame_localIdx = dict()
            for frame_time, local_index in frame_info:
                frame_localIdx[frame_time] = local_index
            id_prev2curr_pred_frame[key] = frame_localIdx
        
        rotation_degrees = np.degrees(np.arctan2(prev2curr_matrix[:3, :3][1, 0], prev2curr_matrix[:3, :3][0, 0]))
        # get the center of the car in the last frame's coordinate
        car_center = get_prev2curr_vectors(np.array((0,0)).reshape(1,1,2), prev2curr_matrix,origin,roi_size,False,False)* roi_size + origin
        car_trajectory.append([car_center.squeeze(), rotation_degrees])

    # sort the id_prev2curr_pred_vectors by label and vec_glb_idx
    id_prev2curr_pred_vectors = {key: id_prev2curr_pred_vectors[key] for key in sorted(id_prev2curr_pred_vectors)}

    # get the x_min, x_max, y_min, y_max for the figure size
    x_min = -roi_size[0] / 2
    x_max = roi_size[0] / 2
    y_min = -roi_size[1] / 2
    y_max = roi_size[1] / 2

    all_points = []
    for vecs in id_prev2curr_pred_vectors.values():
        points = np.concatenate(vecs, axis=0)
        all_points.append(points)
    all_points = np.concatenate(all_points, axis=0)

    x_min = min(x_min, all_points[:,0].min())
    x_max = max(x_max, all_points[:,0].max())
    y_min = min(y_min, all_points[:,1].min())
    y_max = max(y_max, all_points[:,1].max())

    scene_dir = os.path.join(args.out_dir,scene_name)
    os.makedirs(scene_dir,exist_ok=True)

    # if visulize the per frame result
    if args.per_frame_result:
        pred_save_folder = os.path.join(scene_dir, f'gt_merged_per_frame')
        plot_fig_merged_per_frame(len(gt_info_list), car_trajectory, x_min, x_max, y_min, y_max, pred_save_folder, id_prev2curr_pred_vectors, id_prev2curr_pred_frame, args)
        pred_save_folder = os.path.join(scene_dir, f'gt_unmerged_per_frame')
        plot_fig_unmerged_per_frame(len(gt_info_list), car_trajectory, x_min, x_max, y_min, y_max, pred_save_folder, id_prev2curr_pred_vectors, id_prev2curr_pred_frame, args)
    # plot result for across all frames
    pred_save_path = os.path.join(scene_dir, f'gt_unmerged.png')
    plot_fig_unmerged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, id_prev2curr_pred_vectors, args)
    pred_save_path = os.path.join(scene_dir, f'gt_merged.png')
    plot_fig_merged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, id_prev2curr_pred_vectors, args)

    # combine the merged and unmerged images into one plot for comparison
    comb_save_path = os.path.join(scene_dir, f'gt_comb.png')
    image_paths = [os.path.join(scene_dir, f'gt_merged.png'), os.path.join(scene_dir, f'gt_unmerged.png')]
    labels = ['Merged', 'Unmerged']
    combine_images_with_labels(image_paths, labels, comb_save_path)
    print("image saved to : ", comb_save_path)


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

    # load the GT data
    if args.option == "vis-gt": 
        data = mmcv.load(args.data_path)
    # load the prediction data
    elif args.option == "vis-pred":
        with open(args.data_path,'rb') as fp:
            data = pickle.load(fp)

    all_scene_names = sorted(list(scene_name2idx.keys()))

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
            # visualize the GT data
            vis_gt_data(scene_name=scene_name, args=args, dataset=dataset, gt_data=data, origin=origin, roi_size=roi_size)
        elif args.option == "vis-pred":
            # visualize the prediction results
            vis_pred_data(scene_name=scene_name, pred_results=data, origin=origin, roi_size=roi_size, args=args)
        else:
            raise ValueError('Invalid visualization option {}'.format(args.option))


if __name__ == '__main__':
    main() 