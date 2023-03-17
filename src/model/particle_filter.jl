

struct LiquidParticleFilter <: Gen_Compose.AbstractParticleFilter
    particles::Int64
    ess::Float64
    rejuvenation::Union{Function, Nothing}
end


function Gen_Compose.rejuvenate!(proc::LiquidParticleFilter,
                                 state::Gen.ParticleFilterState)
    println("Must rejuvenate")
end


function Gen_Compose.smc_step!(state::Gen.ParticleFilterState,
                               proc::LiquidParticleFilter,
                               query::StaticQuery)
    # Resample before moving on...
    time = query.args[1]
    Gen_Compose.resample!(proc, state)

    # update the state of the particles
    Gen.particle_filter_step!(state, query.args,
                              (UnknownChange(),),
                              query.observations)
    Gen_Compose.rejuvenate!(proc, state)
    return nothing
end



export LiquidParticleFilter, smc_step!
