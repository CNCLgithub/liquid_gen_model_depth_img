module Constants


const BASE_DIR_PATH = joinpath(dirname(dirname(dirname(@__FILE__))))

const BASE_LIB_PATH = joinpath(BASE_DIR_PATH, "library")
const BASE_PY_PATH = joinpath(BASE_DIR_PATH, "ss_julia")
const RESULTS_PATH = joinpath(BASE_DIR_PATH, "out")




export BASE_LIB_PATH, BASE_PY_PATH, RESULTS_PATH

end
