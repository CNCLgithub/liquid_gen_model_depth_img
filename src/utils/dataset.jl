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
    init_frame_num = 1+total_masks
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
	total_slot = Int(round(total_obs/total_masks))
        observations_gen[dir_name] = Vector{Gen.ChoiceMap}(undef, total_slot)
	i = init_frame_num
	decay_rate = [1.0, 0.8, 0.6, 0.4, 0.2]
	d_rate = decay_rate[1:total_masks]
	time_t = 1
	while i <= total_obs
	    k = total_masks
	    flow_pos = []
            for j in i-total_masks+1:i
	        curr_file_path = joinpath(BASE_LIB_PATH, dir_name, "partio", "ParticleData_Fluid_0_"*string(j)*".bgeo")
	        @show curr_file_path
                arguments_pc= ["--bgeo_path", string(curr_file_path),
                        "--output_dir", string(joinpath(BASE_LIB_PATH, dir_name, "depth_map")),
			"--scene_name", string(scene)]
                pos= collect_obs(arguments_pc)
		mask = d_rate[k]
		mask_pos = pos*mask
		if (length(flow_pos) == 0)
		    flow_pos = mask_pos
		else 
		    flow_pos = flow_pos+mask_pos
		end 
		k = k-1
	    end
	    c_map = Gen.choicemap()
            c_map[:kernel => time_t => :fluid_pos] = flow_pos
            observations_gen[dir_name][time_t] = c_map
	    i = i+total_masks
	    time_t = time_t+1
        end
    end

    return init_state_dict, observations_gen

end


export collect_observations

end
