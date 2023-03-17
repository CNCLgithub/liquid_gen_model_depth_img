struct ConfigParams
    scene::String  #(box, obstacle, etc.)
    viscosity_min::Float64
    viscosity_max::Float64
    start_frame::Int64
    total_masks::Int64
end
const default_config_params = ConfigParams("box", -4, 4, Int64(1),Int(3))


struct StateSpace
    log_viscosity::Float64
    viscosity::Float64
end


VISCOSITY_VAR_SMALL = 0.1
VISCOSITY_VAR_LARGE = 0.5 

# observation noise
POS_VAR = 0.1 #originally 0.05, 0.1, 0.15, 0.2

get_var_value_viscosity() = bernoulli(0.9) ? VISCOSITY_VAR_SMALL : VISCOSITY_VAR_LARGE

function printing(t::Int, log_vis::Float64, vis::Float64)
    println("---------------")
    println(t)
    println("log_viscosity: $log_vis, viscosity: $vis")
    return t
end

@gen (static) function sample_init_state(cparam::ConfigParams)
    log_viscosity = @trace(uniform(cparam.viscosity_min-0.1, cparam.viscosity_max+0.1), :viscosity)
    viscosity = 2^log_viscosity
    a = printing(0, log_viscosity, viscosity)
    pos = vec(zeros(250000,1))
    out = @trace(broadcasted_normal(pos, [POS_VAR]), :fluid_pos)

    state::StateSpace = StateSpace(log_viscosity,viscosity)
    return state
end


@gen (static) function fluid_kernel_rand_walk(t::Int, prev_state::StateSpace, cparam::ConfigParams)

    # ---------------------------- LATENT RANDOM WALK ---------------------------------
    prev_log_viscosity = prev_state.log_viscosity
    
    prev_viscosity = 2^prev_log_viscosity

    viscosity_boundary = Float64[cparam.viscosity_min, cparam.viscosity_max]
    new_log_viscosity = @trace(truncated_normal(prev_log_viscosity, get_var_value_viscosity(), viscosity_boundary), :viscosity)
    
    new_viscosity = 2^new_log_viscosity
    a = printing(t, new_log_viscosity, new_viscosity)
    start_frame = cparam.start_frame
    total_masks = cparam.total_masks
    frame = start_frame + t*total_masks
    a = printing(frame, new_log_viscosity, new_viscosity)
    arguments= ["--scene_name", string(cparam.scene),
                "--total_frame", string(frame),
		"--total_masks", string(total_masks),
		"--start_frame", string(start_frame),
                "--viscosity", string(new_viscosity),
                "--prev_viscosity", string(prev_viscosity)]
    bgeo_path, bgeo_file_list = simulate_next_frame(arguments)
    arguments_c = ["--bgeo_path", string(bgeo_path),
		   "--bgeo_file_list", string(bgeo_file_list),
		   "--which_frame", string(frame),
		   "--total_masks", string(total_masks),
		   "--scene_name", string(cparam.scene)]
    flow_depth_map = concatenate_frames(arguments_c)
    out = @trace(broadcasted_normal(flow_depth_map, [POS_VAR]), :fluid_pos)
    state::StateSpace = StateSpace(new_log_viscosity, new_viscosity)
    return state
end


kernel_random_walk = Gen.Unfold(fluid_kernel_rand_walk)


@gen (static) function gm_fluid(T::Int, cparam::ConfigParams)

    init_state = @trace(sample_init_state(cparam), :init_state)
    states = @trace(kernel_random_walk(T, init_state, cparam), :kernel)
    return states
end


export ConfigParams, StateSpace, default_config_params, gm_fluid
