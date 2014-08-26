
export train_local!

function train_local!{T<:Number}(net::HopfieldNetwork, patterns::Array{T,2})
	nn,np = size(patterns) 
	
	@assert false "unimplemented"
	# calc synaptic weight (Local iterative learning method)
	train(net,patterns)
	w = zeros(T,size(net.weight))
	for mu=1:np
		for p=1:nn
			for q=1:nn
				w[p,q] = patterns[p,mu] * conj(patterns[q,mu]) / nn
			end
		end
		net.weight += d*w
	end
end
