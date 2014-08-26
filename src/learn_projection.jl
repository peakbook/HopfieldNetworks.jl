
export train_proj!
function train_proj!{T<:Number}(net::HopfieldNetwork, patterns::Array{T,2})
	nn,np = size(patterns) 

	# calc synaptic weight (Projection learning rule)
	Qinv=pinv(patterns' * patterns/nn)
	for p in 1:nn
		for q in p+1:nn
			s = zero(T) 
			for mu in 1:np
				for nu in 1:np
					s += patterns[p, mu] * Qinv[mu,nu] * conj(patterns[q, nu]) 
				end
			end
			s /= nn
			net.weight[p, q] = s
			net.weight[q, p] = conj(s)
		end
	end

	return
end

