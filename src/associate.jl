
export associate!

function update_async!(net::HopfieldNetwork, activatefunc::Function)
	for i in randperm(length(net.state))
		net.state[i] = activatefunc(dot(net.weight[:, i], net.state))
	end
	return
end

function update_sync!(net::HopfieldNetwork, activatefunc::Function)
 	state = copy(net.state)
	for i=1:length(state)
		net.state[i] = activatefunc(dot(net.weight[:, i], state))
	end
	return
end

function associate!{T <: Number}(net::HopfieldNetwork{T}, pattern::AbstractVector{T}, activatefunc::Function, iterations::Integer = 1_000, sync::Bool = false)
	copy!(net.state, pattern)
	associate!(net,activatefunc,iterations,sync)
end

function associate!(net::HopfieldNetwork, activatefunc::Function, iterations::Integer = 1_000, sync::Bool = false)
	ite = iterations
	state = copy(net.state)

	if sync
		state2 = copy(net.state)
		for i in 1:iterations
			update_sync!(net,activatefunc)
			if state2 == net.state
				ite = i
				break
			end
			copy!(state2,state)
			copy!(state,net.state)
		end
	else
		for i in 1:iterations
			update_async!(net,activatefunc)
			if state == net.state
				ite = i
				break
			end
			copy!(state,net.state)
		end
	end

	return ite
end

