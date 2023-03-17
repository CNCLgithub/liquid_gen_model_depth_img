module OBJObservations

using ..PyTalker
using ..OBJUtils
using ..Constants
using Formatting
using Statistics
using Distances


get_init_obs(obj_path::String, t_interval::Float64) = get_obj_data(obj_path, nothing, t_interval)


function get_curr_obs(prev_file_path::String, curr_file_path::String, time_interval::Float64)
    """
    prev_file_path : path to the obj file in the previous time step
    curr_file_path : path to the obj file in the current time step
    """
    return get_obj_data(curr_file_path, prev_file_path, time_interval)
end


all_zeros(input_arr) = all(k -> k==0.0, input_arr)
all_greater(input_arr) = any(k -> k>0.1, input_arr)


function mat_to_vec_of_vec(A::AbstractMatrix{T}) where T
    m, n = size(A)
    B = Vector{T}[Vector{T}(undef, n) for _ in 1:m]
    for i=1:m
       B[i] .= A[i, :]
    end
    return B
end


function get_coherence_value(cloth_vel::Array{Array{Float64, 1}, 1})
    coherernce_values = Float64[]

    all_x_values = Float64[]
    all_y_values = Float64[]
    all_z_values = Float64[]

    total_rows_cols = Int64(sqrt(length(cloth_vel)))
    sub_matrix_size = 5

    for i=1:length(cloth_vel)
        push!(all_x_values, cloth_vel[i][1])
        push!(all_y_values, cloth_vel[i][2])
        push!(all_z_values, cloth_vel[i][3])
    end

    vel_mean = var(all_x_values) + var(all_y_values) + var(all_z_values)

    cloth_vel_mat = transpose(reshape(cloth_vel, (total_rows_cols, total_rows_cols)))

    for i=1:sub_matrix_size:total_rows_cols - 1
        for j=1:sub_matrix_size:total_rows_cols - 1

            flattened_sub_mat = cloth_vel_mat[i:min(i + sub_matrix_size - 1, total_rows_cols),
                                              j:min(j + sub_matrix_size - 1, total_rows_cols)]
            flattened_sub_mat = vec(flattened_sub_mat)

            x_values = Float64[]
            y_values = Float64[]
            z_values = Float64[]
            for k=1:length(flattened_sub_mat)
                x_val = flattened_sub_mat[k][1]
                y_val = flattened_sub_mat[k][2]
                z_val = flattened_sub_mat[k][3]
                push!(x_values, flattened_sub_mat[k][1])
                push!(y_values, flattened_sub_mat[k][2])
                push!(z_values, flattened_sub_mat[k][3])
                push!(coherernce_values, sqrt(x_val^2 + y_val^2 + z_val^2))
            end
#             push!(coherernce_values, (var(x_values) + var(y_values) + var(z_values)) / vel_mean)
        end
    end
    return coherernce_values
end


function get_coherence_value(cloth_vel::Array{Float64, 1},
                             total_points_per_mask::Int64, total_masks::Int64)

    total_videos = Int64(length(cloth_vel) / (total_points_per_mask * 3))

    coherernce_values = Float64[]

    cloth_vel_mat = transpose(reshape(cloth_vel, (3, total_points_per_mask * total_videos)))
    for i=0:total_videos - 1
        first_index = i * total_points_per_mask + 1
        last_index = i * total_points_per_mask + total_points_per_mask

        cloth_vel_mat_per_time = deepcopy(mat_to_vec_of_vec(cloth_vel_mat[first_index:last_index, :]))

        coherence = get_coherence_value(cloth_vel_mat_per_time)
        coherernce_values = [coherernce_values; coherence]
    end

    return coherernce_values
end


function get_final_flattened_obs(prev_flattened_masked::Array{Float64, 1},
                                 curr_flattened_cloth::Array{Float64, 1},
                                 total_points_per_mask::Int64,
                                 total_masks::Int64,
                                 time_step_num::Int64)

    total_points_per_mask = total_points_per_mask * 3

    if total_masks == 0 || time_step_num == 1
        return deepcopy(curr_flattened_cloth)
    end

    returned = deepcopy(curr_flattened_cloth)
    returned = [returned; prev_flattened_masked]
    if length(prev_flattened_masked) < total_points_per_mask * (total_masks + 1)
        return deepcopy(returned)
    end

    return deepcopy(returned[1:end - total_points_per_mask])
end


function get_simulated_data(cloth_positions::Array{Array{Float64, 1}, 1},
                            cloth_velocities::Array{Array{Float64, 1}, 1},
                            object_positions::Array{Array{Float64, 1}, 1},
                            object_velocities::Array{Array{Float64, 1}, 1},
                            mass::Float64,
                            stiffness::Float64,
                            external_force::Float64,
                            sim_num::Int64,
                            time_step_num::Int64,
                            time_interval::Float64,
                            extention::String)
    """
    positions       : the array of all position vectors (vertices)
    velocities      : the array of all velocity vectors
    sim_num         : the simulation number designating the scenario being played
    time_step_num   : the time step to be queried
    """
    println("simulating..." * ": t = " * string(time_step_num) * ", mass = " * string(mass) * ", stiffness = " * string(stiffness))

    arguments = ["--scenes_str", string(ScenarioType(sim_num)),
                 "--input_t_frame", string(time_step_num),
                 "--cloth_position", string(cloth_positions),
                 "--cloth_velocity", string(cloth_velocities),
                 "--object_position", string(object_positions),
                 "--object_velocity", string(object_velocities),
                 "--mass", format(mass),
                 "--bstiff_float", format(stiffness),
                 "--shstiff", format(stiffness),
                 "--ststiff", format(stiffness),
                 "--flex_output_root_path", string("experiments/simulation/" * extention * "/trials/")]

    simulate_next_frame_flex(arguments)

    sim_path = joinpath(BASE_PY_PATH, "experiments/simulation/" * extention)
    obj_path_prev = joinpath(sim_path, "trials_cloth_0.obj")
    obj_path_curr = joinpath(sim_path, "trials_cloth_1.obj")

    obj_data = get_obj_data(obj_path_curr, obj_path_prev, time_interval)
    if sim_num == 3 && time_step_num < BALL_SCENARIO_START_INDEX - 1
        obj_data.object_positions = deepcopy(object_positions)
        obj_data.object_velocities = deepcopy(object_velocities)
    end

    return obj_data
end


@enum ScenarioType wind=1 drape ball rotate


export ScenarioType, wind, drape, ball, rotate, get_init_obs, get_curr_obs, get_masked_obs,
       get_coherence_value

end