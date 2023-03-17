module FluidGen

using Gen
using Statistics
using LinearAlgebra
using Gen_Compose
using Reexport

include("./utils/truncated_normal.jl")

include("./utils/constants.jl")
@reexport using .Constants

include("./controller/py_talker.jl")
@reexport using .PyTalker

include("./utils/dataset.jl")
@reexport using .DatasetCollector

include("./model/particle_filter.jl")
include("./model/generative_model.jl")


__init__() = @load_generated_functions


end
