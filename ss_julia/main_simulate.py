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


"""
##########################################################################################
Script Name                : simulator.py
Description                : This script runs fluid simulation
Usage                      : 1) Create new scene configuration file
                             2) Set parameters in scene configuration file
                             3) Run simulation
Output                     : [outputFolder]/*.obj for rigid bodies, [outputFolder]/*.bgeo
                                 for fluid
Date                       : 12/08/2021
Author                     : Yuting Zhang
##########################################################################################
"""
"""
directory setup:
|-- OUTPUT_DIR/
    |-- simulation_$date
        |-- configuration
            |-- scene_config1.json
            |-- scene_config2.json
            |-- scene_config3.json
            |-- scene_config4.json
        |-- output
            |-- $scene_name_$vis
                |-- obj
                |-- state
                |-- partio
                |-- output (bgeo interpolation)
            |-- $scene_name_$vis
            |-- $scene_name_$vis
            |-- $scene_name_$vis
"""
# Set up working directory
#PATH_TO_SIMULATOR = "/home/yz932/project/SPlisHSPlasH_tmp/bin"
#OUTPUT_DIR = "/home/yz932/project/SPlisHSPlasH/bin/output/"
#BASE_SCENE_DIR = "/home/yz932/project/SPlisHSPlasH/data/SceneConfig/"
cwd = str(os.getcwd())

PATH_TO_SIMULATOR = os.path.join(os.path.dirname(os.path.dirname(cwd)),"SPlisHSPlasH_tmp/bin")
OUTPUT_DIR = "/home/yz932/project/SPlisHSPlasH/bin/output/"
BASE_SCENE_DIR = "/home/yz932/project/SPlisHSPlasH/data/SceneConfig/"

SIMULATION_CMD = "./SPHSimulator"
FPS = 25
# Set up parser
parser = argparse.ArgumentParser(description='This script use SPlisHSPlasH to simulate fluid dynamics.')
# Experiments
parser.add_argument('--scene_name', type=str, default="box", help='Scene name for simulation.')
parser.add_argument('--total_frame', type=str, default=5, help='Total number of frames to simulate.')
parser.add_argument('--total_masks', type=str, default=5, help='total_masks')
parser.add_argument('--start_frame', type=str, default=5, help='start_frame')
parser.add_argument('--viscosity', type=str, help='Input viscosity.')
parser.add_argument('--prev_viscosity', type=str, default="-1", help="Previous viscosity.")
# Set up environment
# Get arguments
args = parser.parse_args(sys.argv)
total_frame = ast.literal_eval(args.total_frame)
prev_vis = ast.literal_eval(args.prev_viscosity)
vis = ast.literal_eval(args.viscosity)
total_masks= ast.literal_eval(args.total_masks)
start_frame = ast.literal_eval(args.start_frame)
#stop_at = float(total_frame/FPS)
stop_at = float((total_frame/FPS)-0.01)

scene_name = args.scene_name
path_to_scene_file = BASE_SCENE_DIR + scene_name + ".json"
#state file path
if str(total_frame) != str(total_masks+start_frame):
    # Create simulation folder
    #get datetime
    today = date.today()
    d = today.strftime("%m%y")
    #create dir for scene configuration and output
    dir = OUTPUT_DIR+"simulation_%d/"%int(d)
    #check whether dir exists, if not, make one
    if not os.path.exists(dir):
        os.mkdir(dir)
    #check whether configuration exists, if not, make one
    configuration = dir+"configuration/"
    if not os.path.exists(configuration):
        os.mkdir(configuration) #make /$simulation_date/configuration
    #check whether output exists, if not, make one
    output = dir+"output/"
    if not os.path.exists(output):
        os.mkdir(output) #make /$simulation_date/output
    state_file_path = os.path.join(dir,"output",scene_name+"_"+str(prev_vis).replace(".","-"),"state")
    #state file to be loaded
    #state_file = sorted(os.listdir(state_file_path))[0]
    state_file = sorted(os.listdir(state_file_path))[-2]
else:
    # Create simulation folder
    #get datetime
    today = date.today()
    d = today.strftime("%m%y")
    #create dir for scene configuration and output
    # global sim_dir
    sim_dir = "simulation_%d/"%int(d)
    dir = OUTPUT_DIR+sim_dir
    #check whether dir exists, if not, make one
    if not os.path.exists(dir):
        os.mkdir(dir)
    #check whether configuration exists, if not, make one
    configuration = dir+"configuration/"
    if not os.path.exists(configuration):
        os.mkdir(configuration) #make /$simulation_date/configuration
    #check whether output exists, if not, make one
    output = dir+"output/"
    if not os.path.exists(output):
        os.mkdir(output) #make /$simulation_date/output

    #copy
    #file name
cmd_cp = ["cp",path_to_scene_file,configuration+scene_name+"_"+str(vis).replace(".","-")+".json"]
os.system(" ".join(cmd_cp))
working_scene_file = configuration+scene_name+"_"+str(vis).replace(".","-")+".json"
    #write scene configuration file
with open(working_scene_file, 'r') as f:
    data = json.load(f)
    data["Materials"][0]["viscosity"] = float(vis) #set viscosity
with open(working_scene_file, 'w') as f:
    json.dump(data,f)

    #create /$simulation_date/output/$scene_name+$vis dir for simulation output
# global output_dir
output_dir = output+scene_name+"_"+str(vis).replace(".","-")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    # Set up simulation command
cwd = os.getcwd()
os.chdir(PATH_TO_SIMULATOR)
os.getcwd()
if str(total_frame) == str(total_masks+start_frame):
    sim_cmd = ["MESA_GL_VERSION_OVERRIDE=3.3", SIMULATION_CMD,
        working_scene_file, "--output-dir", output_dir, "--no-gui",
        "--no-initial-pause", "--stopAt", str(stop_at), ">", os.path.join(dir, "tmp.txt")]
else:
    state = os.path.join(state_file_path,state_file)
    #print(state)
    sim_cmd = ["MESA_GL_VERSION_OVERRIDE=3.3", SIMULATION_CMD,
    working_scene_file, "--output-dir", output_dir, "--no-gui",
    "--no-initial-pause", "--stopAt", str(stop_at), "--state-file", state,
    "--param Fluid:viscosity:", str(vis), ">", os.path.join(dir, "tmp.txt")]

os.system(" ".join(sim_cmd))
# Find the targeted bgeo file
global bgeo_file_path,bgeo_file
bgeo_file_path = output+scene_name+"_"+str(vis).replace(".","-")+"/partio/"
#bgeo_file = sorted(os.listdir(bgeo_file_path))[0]
bgeo_file = sorted(os.listdir(bgeo_file_path))
# print(globals()['sim_dir'],globals()['bgeo_file'])


