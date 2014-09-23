module HopfieldNetworks

using Lazy
using Quaternions

include("base.jl")
include("learn_hebb.jl")
include("learn_projection.jl")
include("learn_local.jl")
include("associate.jl")
include("associate_cue.jl")
include("activatefuncs.jl")
include("overlap.jl")

end # module
