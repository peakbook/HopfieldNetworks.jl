
export HopfieldNetwork

type HopfieldNetwork{T<:Number}
	state::Array{T,1}
	weight::Array{T,2}
	n::Integer
	typ::Type
	function HopfieldNetwork(n::Integer)
		state = zeros(T,n)
		Weight = zeros(T,(n,n))
		new(state,Weight,n,T)
	end
end

function Base.show(io::IO, net::HopfieldNetwork)
	@printf io "HopfieldNetwork(type: %s, dim: %d)" string(net.typ) net.n
end

