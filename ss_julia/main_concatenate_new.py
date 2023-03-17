import os, subprocess, re, sys, ast
import json
import numpy as np
import random
import argparse
import math
from shutil import copyfile
from datetime import date
import time
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from ast import literal_eval
import meshio
import open3d as o3d
from visualization import DepthMapOpen3D
import scipy


# Set up parser
parser = argparse.ArgumentParser(description='This script use SPlisHSPlasH to simulate fluid dynamics.')
# Experiments
parser.add_argument('--bgeo_path', type=str, default="box", help='bgeo_path.')
parser.add_argument('--bgeo_file_list', type=str, default=5, help='bgeo_file_list.')
parser.add_argument('--which_frame', type=str, help="which frame?")
parser.add_argument('--total_masks',type=str,help="total masks?")
parser.add_argument('--scene_name',type=str,help="which scene?")
# Set up environment
# Get arguments
args = parser.parse_args(sys.argv)
bgeo_list = ast.literal_eval(args.bgeo_file_list)
bgeo_path = args.bgeo_path
which_frame = ast.literal_eval(args.which_frame)
total_masks = ast.literal_eval(args.total_masks)
scene_name = args.scene_name
first_batch = len(bgeo_list)
if first_batch == 4:
    bgeo_list.pop(0)
def numpy_from_bgeo(path):
    import partio
    import numpy as np
    p = partio.read(path)
    pos = p.attributeInfo('position')
    vel = p.attributeInfo('velocity')
    ida = p.attributeInfo('trackid')  # old format
    if ida is None:
        ida = p.attributeInfo('id')  # new format after splishsplash update
    n = p.numParticles()
    pos_arr = np.empty((n, pos.count))
    for i in range(n):
        pos_arr[i] = p.get(pos, i)
    vel_arr = None
    if not vel is None:
        vel_arr = np.empty((n, vel.count))
        for i in range(n):
            vel_arr[i] = p.get(vel, i)
    if not ida is None:
        id_arr = np.empty((n,), dtype=np.int64)
        for i in range(n):
            id_arr[i] = p.get(ida, i)[0]
        s = np.argsort(id_arr)
        result = [pos_arr[s]]
        if not vel is None:
            result.append(vel_arr[s])
    else:
        result = [pos_arr, vel_arr]
    return tuple(result)


def get_depth_map(pos_data,
                  scene,
                  path, 
                  output_dir,
                  w=200,
                  h=200):
    """
    Global params
    points_or_mesh: mesh | points
    """
    # if blender_cam_config_path is None:
    #     blender_cam_config_path = os.path.join(os.path.dirname(__file__), '..', 'blender_cam.json')
    #     with open(blender_cam_config_path) as f:
    #         blender_cam = json.load(f)

    blender_cam = {"box": {'intrinsic': [[694.44446, 0.0, 250.0], [0.0, 694.44446, 250.0], [0.0, 0.0, 1.0]], 'extrinsic': [[-0.77153, -0.63605, -0.01366, 0.03376], [-0.14014, 0.19086, -0.97156, 0.58041], [0.62057, -0.74767, -0.23639, 13.54276], [0.0, 0.0, 0.0, 1.0]], 'pos': [-8.29685, 10.03624, 3.76569], 'viewup': [0.14014, -0.19086, 0.97156], 'focal_dis': 10, 'dir': [0.62057, -0.74767, -0.23639], 'clipping_range': [0.1, 1000.0], 'fluid_rot': [90, 0, 0], 'res': [500, 500], 'view_angle': 39.59776},
"boxwithahole": {'intrinsic': [[694.44446, 0.0, 250.0], [0.0, 694.44446, 250.0], [0.0, 0.0, 1.0]], 'extrinsic': [[-0.76154, -0.64812, 0.0, 0.11368], [-0.12145, 0.1427, -0.98229, 0.4313], [0.63664, -0.74805, -0.18738, 10.34964], [0.0, 0.0, 0.0, 1.0]], 'pos': [-6.45004, 7.75418, 2.363], 'viewup': [0.12145, -0.1427, 0.98229], 'focal_dis': 10, 'dir': [0.63664, -0.74805, -0.18738], 'clipping_range': [0.1, 1000.0], 'fluid_rot': [0, 0, 90], 'res': [500, 500], 'view_angle': 39.59776},
"motor": {'intrinsic': [[694.44446, 0.0, 250.0], [0.0, 694.44446, 250.0], [0.0, 0.0, 1.0]], 'extrinsic': [[-0.99938, -0.03258, -0.01363, 0.03289], [0.00308, 0.30388, -0.95271, 0.25408], [0.03518, -0.95215, -0.30359, 4.78455], [0.0, 0.0, 0.0, 1.0]], 'pos': [-0.13624, 4.47949, 1.69505], 'viewup': [-0.00308, -0.30388, 0.95271], 'focal_dis': 10, 'dir': [0.03518, -0.95215, -0.30359], 'clipping_range': [0.1, 1000.0], 'fluid_rot': [90, 0, 90], 'res': [500, 500], 'view_angle': 39.59776},
"oneobject": {'intrinsic': [[694.44446, 0.0, 250.0], [0.0, 694.44446, 250.0], [0.0, 0.0, 1.0]], 'extrinsic': [[-0.99997, -0.00728, 0.00195, -0.00361], [-0.0064, 0.68473, -0.72876, 0.02136], [0.00397, -0.72876, -0.68476, 12.38485], [0.0, 0.0, 0.0, 1.0]], 'pos': [-0.05262, 9.01088, 8.49625], 'viewup': [0.0064, -0.68473, 0.72876], 'focal_dis': 10, 'dir': [0.00397, -0.72876, -0.68476], 'clipping_range': [0.1, 1000.0], 'fluid_rot': [90, 0, 90], 'res': [500, 500], 'view_angle': 39.59776},
"wall": {'intrinsic': [[694.44446, 0.0, 250.0], [0.0, 694.44446, 250.0], [0.0, 0.0, 1.0]], 'extrinsic': [[-0.6743, -0.73845, -0.00104, -0.38684], [-0.10787, 0.09989, -0.98913, -0.13403], [0.73053, -0.66687, -0.14702, 22.26198], [0.0, 0.0, 0.0, 1.0]], 'pos': [-16.53838, 14.57347, 3.13993], 'viewup': [0.10787, -0.09989, 0.98913], 'focal_dis': 10, 'dir': [0.73053, -0.66687, -0.14702], 'clipping_range': [0.1, 1000.0], 'fluid_rot': [90, 0, 180], 'res': [500, 500], 'view_angle': 39.59776},
"obstacle": {'intrinsic': [[694.44446, 0.0, 250.0], [0.0, 694.44446, 250.0], [0.0, 0.0, 1.0]], 'extrinsic': [[-0.89453, -0.44677, -0.01428, 0.40286], [0.02779, -0.02371, -0.99933, 0.95459], [0.44614, -0.89433, 0.03362, 5.89928], [0.0, 0.0, 0.0, 1.0]], 'pos': [-2.29804, 5.47854, 0.76136], 'viewup': [-0.02779, 0.02371, 0.99933], 'focal_dis': 10, 'dir': [0.44613, -0.89433, 0.03362], 'clipping_range': [0.1, 1000.0], 'fluid_rot': [90, 0, 180], 'res': [500, 500], 'view_angle': 39.59776}}

    from scipy.spatial.transform import Rotation as R
    import open3d as o3d
    import numpy as np
    from visualization import DepthMapOpen3D
    import os, sys, argparse, ast
    import matplotlib.pyplot as plt
    cur_scene = scene
    print("*************************")
    print(cur_scene)
    rot = R.from_euler('xyz', blender_cam[scene]['fluid_rot'], degrees=True)
    rot = rot.as_matrix()
    points = np.asarray(pos_data)
    rotated_points = np.dot(rot, points.transpose())
    rotated_points = rotated_points.transpose() #has to make the [1] position in shape to 3
        ### ========== Point clouds ================== ###
    pcd = o3d.geometry.PointCloud() #initiate point cloud
    pcd.points = o3d.utility.Vector3dVector(rotated_points) #convert rotated_points from matrix to point cloud

        # set up viewer
        # cur_scene = scene # extract it from the path passed in
    WIDTH = blender_cam[cur_scene]['res'][0]
    HEIGHT = blender_cam[cur_scene]['res'][1]
    INTRINSIC = np.array(blender_cam[cur_scene]['intrinsic'])
    EXTRINSIC = np.array(blender_cam[cur_scene]['extrinsic'])
    window_visible = False
    vis = DepthMapOpen3D(img_width=WIDTH, img_height=HEIGHT, visible=window_visible)
    vis.add_geometry(pcd)
    vis.update_view_point(INTRINSIC, EXTRINSIC)

    depth = vis.capture_depth_float_buffer(show=False)
    image = vis.capture_screen_float_buffer(show=False)

        # --------------save------------------------#
    name = path.split("/")[-1]
    name = name.replace(".bgeo","")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    vis.capture_screen_image(os.path.join(output_dir,name+"_screen_image.png"))
    vis.capture_depth_image(os.path.join(output_dir,name+"_depth_image.png"))
    depth = np.array(depth).astype(np.float64)
    return depth


#depth_output_dir = os.path.join(os.path.dirname(os.path.dirname(bgeo_path)), "depth")
depth_output_dir = os.path.join("/home/yz932/project/liquid_gen_model","depth")
if not os.path.exists(depth_output_dir):
    os.mkdir(depth_output_dir)
decay_rate = [1.0, 0.8, 0.6, 0.4, 0.2]
decay_r = decay_rate[0:total_masks]
#flow = []
for i, bgeo in enumerate(bgeo_list):
    b = "ParticleData_Fluid_0_"+str(which_frame)+".bgeo"
    path = os.path.join(bgeo_path,b)
    result = numpy_from_bgeo(path)
    pos = result[0]
    vel = result[1]
    depth = get_depth_map(pos_data=pos, scene=scene_name, path = path,  output_dir=depth_output_dir) 
    if i == 0:
        flow = depth
    else:
        flow += depth * decay_r[i]
    #depth_list = list(depth)
    #mask = decay_r[i]
    #flow_depth = [x*mask for x in depth_list]
    #flow_depth_array = np.array(flow_depth)
    #flow.append(flow_depth_array)
    #i+=1
    which_frame -= 1

np.savetxt(os.path.join("/home/yz932/project/liquid_gen_model/flow",str(which_frame)+".txt"),flow,fmt='%d')
global flow_depth_map
#flow_d = np.sum(np.array(flow),axis=0)
#flow_depth_map = flow_d.flatten()
flow_depth_map = flow.flatten()

