
export train!

# train core 1 {{{
function train_core1(p::Real,q::Real)
	return p * q 
end

function train_core1{T<:Real}(p::Complex{T},q::Complex{T})
	return p * conj(q)
end

function train_core1{T<:Real}(p::Quaternion{T},q::Quaternion{T})
	return p * conj(q)
end
# }}}

# train core 2 {{{
function train_core2(x::Real,nn::Real)
	return x/nn
end

function train_core2{T<:Real}(z::Complex{T},nn::Real)
	return z/(2.0*nn)
end

function train_core2{T<:Real}(q::Quaternion{T},nn::Real)
	return q/(4.0*nn)
end
# }}}

function train!{T<:Number}(net::HopfieldNetwork, patterns::Array{T,2})
	nn,np = size(patterns) 

	# calc synaptic weight (Hebbian learning rule)
	for p in 1:nn
		for q in p+1:nn
			s = zero(T) 
			for mu in 1:np
				s += train_core1(patterns[p, mu], patterns[q, mu])
			end
			s = train_core2(s,nn)
			net.weight[p, q] = s
			net.weight[q, p] = conj(s)
		end
	end

	return
end


function train_core{T<:Real}(patterns::AbstractArray{T},nn::Real)
	tmp = BLAS.syrk('U','T',1.0/nn, patterns')
	for i=1:nn
		tmp[i,i] = zero(T)
		for j=i+1:nn
			tmp[j,i] = tmp[i,j]
		end
	end

	return tmp
end

function train_fast!{T <: Number}(net::HopfieldNetwork, patterns::Array{T})
	nn = length(net.state) # num of neurons
	np = size(patterns, 2) # num of patterns

	# calc synaptic weight
	net.Weight = train_core(patterns,nn)
	return
end

