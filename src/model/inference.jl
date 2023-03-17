using Gen
using Dates
using Gen_Compose


TOTAL_INFER_ITERATIONS = 1


function extract_vis(trace::Gen.Trace)
    t = first(Gen.get_args(trace))
    states = Gen.get_retval(trace)
    reshape([states[t].log_viscosity], (1,1,1))
end


function run(num_particles, init_state_dict, observations_dict, total_masks)

    # scope for parallelization
    for dir_path in collect(keys(observations_dict))
        for iter in 1:TOTAL_INFER_ITERATIONS
            println("probing dir: " * string(dir_path))

            constraints = Gen.choicemap()
            constraints[:init_state => :fluid_pos] = vec(zeros(250000,1))
            observations = observations_dict[dir_path]
	    @show length(observations)
            latents = LatentMap(Dict(
                :viscosity => extract_vis
            ))
	    dir_l = splitdir(dir_path)
	    idx = length(dir_l)
	    dir = dir_l[idx]
	    scene_name, visco = split(dir,"_") 
            c_params = ConfigParams(string(scene_name), -4.5, 4.5, Int64(1), Int64(total_masks))
            query = Gen_Compose.SequentialQuery(latents,
                                                gm_fluid,
                                                (0, c_params),
                                                constraints,
                                                [(i, c_params) for i in 1:length(observations)],
                                                observations)

            particle_filter = LiquidParticleFilter(num_particles, num_particles/2, nothing)


            results_filename = "result_" * string(dir) * "_" * string(iter) * ".h5" * "-" * Dates.format(now(), "mm-dd-HH")
            results = sequential_monte_carlo(particle_filter, query,
                                             buffer_size=length(observations),
                                             path=joinpath(RESULTS_PATH, results_filename))
        end
    end
end
