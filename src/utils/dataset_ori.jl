module DatasetCollector

using Gen
using ..PyTalker
using ..Constants


function collect_observations(specific_dir_list::Array{String,1}=String[], total_masks::Int64=0)

    obs_bgeo_filenames = Dict()
    if isempty(specific_dir_list)
        error("not implemented")
    else
        for s_dir in specific_dir_list
            files = readdir(joinpath(s_dir,"partio"))
            obs_bgeo_filenames[s_dir] = sort!(files)
        end
    end


    # record observations in Gen using the captured OBJ files
    init_frame_num = 1
    init_state_dict = Dict()
    observations_gen = Dict()
    for dir_name in collect(keys(obs_bgeo_filenames))
        @show dir_name
        dir_l = splitdir(dir_name)
        idx = length(dir_l)
        dir = dir_l[idx]
        scene, visco = split(dir,"_")
        @show scene, visco
        total_obs = length(obs_bgeo_filenames[dir_name])
        observations_gen[dir_name] = Vector{Gen.ChoiceMap}(undef, total_obs)
        for i in init_frame_num:total_obs
            time_t = i
            curr_file_path = joinpath(BASE_LIB_PATH, dir_name, "partio", "ParticleData_Fluid_0_"*string(i)*".bgeo")
	    @show curr_file_path
            arguments_pc= ["--bgeo_path", string(curr_file_path),
                        "--output_dir", string(joinpath(BASE_LIB_PATH, dir_name, "depth_map")),
			"--scene_name", string(scene)]
            pos, vel = collect_obs(arguments_pc)
            c_map = Gen.choicemap()
            c_map[:kernel => time_t => :fluid_pos] = pos
	    c_map[:kernel => time_t => :fluid_vel] = vel
            observations_gen[dir_name][time_t] = c_map
        end
    end

    return init_state_dict, observations_gen

end


export collect_observations

end
