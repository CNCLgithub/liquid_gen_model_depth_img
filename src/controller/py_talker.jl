module PyTalker

using ..Constants


using PyCall


PY_DIR_PATH = BASE_PY_PATH
PY_FILENAME = joinpath(BASE_PY_PATH, "main_simulate.py")
COLLECT = joinpath(BASE_PY_PATH, "observations.py")
CONCATENAT = joinpath(BASE_PY_PATH, "main_concatenate.py")

function simulate_next_frame(arguments,
                                  py_dir_path::String=PY_DIR_PATH,
                                  py_file_path::String=PY_FILENAME)
    py"""
    def call_simulator(arguments, py_dir, py_file_path):
        import sys, os
        import numpy as np
        os.chdir(py_dir)
        if py_dir not in sys.path:
            sys.path.append(py_dir)
        sys.argv = arguments
        exec(open(py_file_path).read())
        return (globals()['bgeo_file_path'], globals()['bgeo_file'])
    """
    call_simulator = py"call_simulator"
    bgeo_file_path,bgeo_file = call_simulator(arguments, py_dir_path, py_file_path)
    return bgeo_file_path::String, bgeo_file::Vector{String}
end

function concatenate_frames(arguments,
                            py_dir_path::String=PY_DIR_PATH,
                            py_file_path::String=CONCATENAT)
    py"""
    def call_c(arguments, py_dir, py_file_path):
        import sys, os
        import numpy as np
        os.chdir(py_dir)
        if py_dir not in sys.path:
            sys.path.append(py_dir)
        sys.argv = arguments
        exec(open(py_file_path).read())
        return (globals()['flow_depth_map'])
    """
    call_c = py"call_c"
    flow_depth_map = call_c(arguments, py_dir_path, py_file_path)
    return flow_depth_map::Vector{Float64}
end

function collect_obs(arguments,
                          py_dir_path::String=PY_DIR_PATH,
                          py_file_path::String=COLLECT)
    py"""
    def collect_observation(arguments, py_dir, py_file_path):
        import sys, os
        import numpy as np
        os.chdir(py_dir)
        if py_dir not in sys.path:
            sys.path.append(py_dir)
        sys.argv = arguments
        exec(open(py_file_path).read())
        return (globals()['depth_map'])
    """
    collect_observation = py"collect_observation"
    depth_map = collect_observation(arguments, py_dir_path, py_file_path)
    return depth_map::Vector{Float64}
end

export simulate_next_frame, concatenate_frames, collect_obs

end
