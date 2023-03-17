using JLD2
using Gadfly
using ColorSchemes


OUT_PATH = joinpath(dirname(dirname(dirname(@__FILE__))), "out")


function extract_chain(jld_data)
    weighted = []
    unweighted = []
    log_scores = []
    ml_est = []
    for t in keys(jld_data)
        for attrib in keys(jld_data[t])
            state = jld_data[t]
            if attrib == "weighted"
                push!(weighted, state[attrib])
            elseif attrib == "unweighted"
                push!(unweighted, state[attrib])
            elseif attrib == "log_scores"
                push!(log_scores, state[attrib])
            elseif attrib == "ml_est"
                push!(ml_est, state[attrib])
            end
        end
    end
    weighted = merge(vcat, weighted...)
    unweighted = merge(vcat, unweighted...)
    log_scores = vcat(log_scores...)
    extracts = Dict("weighted" => weighted,
                    "unweighted" => unweighted,
                    "log_scores" => log_scores,
                    "ml_est" => ml_est)
    return extracts
end


function visualize(jld_data)
    extracted_values = extract_chain(jld_data)

    latents = keys(extracted_values["unweighted"])
    total_t = size(extracted_values["log_scores"])[1]
    particles = size(extracted_values["log_scores"])[2]

    for t in 1:total_t
        p = plot(x=extracted_values["unweighted"][:cloth_mass][t,:],
                 y=extracted_values["unweighted"][:cloth_stiffness][t,:],
                 color=extracted_values["log_scores"][t,:],
                 Geom.rectbin,
                 Scale.ContinuousColorScale(p -> get(ColorSchemes.sunset, p)));
        Gadfly.draw(SVG(joinpath(OUT_PATH, "t_" * string(t) * ".svg"), 5inch, 5inch), p)
    end
end


file = jldopen(joinpath(OUT_PATH, "test_results.h5"), "r")
visualize(file)


export visualize
