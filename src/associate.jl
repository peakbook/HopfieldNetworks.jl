
export associate!

function update_async!(net::HopfieldNetwork, activatefunc::Function)
	ord = shuffle!([1:length(net.state)])
	for i in ord
		net.state[i] = activatefunc(dot(net.weight[:, i], net.state))
	end
	return
end

function associate!{T <: Number}(net::HopfieldNetwork, pattern::AbstractVector{T}, activatefunc::Function, iterations::Integer = 1_000)
	copy!(net.state, pattern)
	associate!(net,activatefunc,iterations)
end

function associate!(net::HopfieldNetwork, activatefunc::Function, iterations::Integer = 1_000)
	ite = iterations
	state = copy(net.state)
	for i in 1:iterations
		update_async!(net,activatefunc)
		if state == net.state
			ite = i
			break
		end
		copy!(state,net.state)
	end
	return ite
end

