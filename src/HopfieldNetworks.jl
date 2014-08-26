module HopfieldNetworks

using Quaternions

include("base.jl")
include("learn_hebb.jl")
include("learn_projection.jl")
include("learn_local.jl")
include("associate.jl")
include("activatefuncs.jl")

end # module
