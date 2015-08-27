
function prob(beta::Real, h::Real)
	return 1.0/(1.0+exp(-beta*h))    # beta = 1/T
end

function stochastic_sign{T<:Real}(p::T)
    return p > rand() ? one(T) : -one(T)
end

function activatefunc(beta::Float64, h::Float64)
    return stochastic_sign(prob(beta,h))
end

function activatefunc(beta::Float64, h::Complex{Float64})
    r_val = stochastic_sign(prob(beta,real(h)))
	i_val = stochastic_sign(prob(beta,imag(h)))
	return complex(r_val, i_val)
end

function activatefunc(beta::Float64, h::Quaternion{Float64})
    r_val = stochastic_sign(prob(beta,real(h)))
	i_val = stochastic_sign(prob(beta,imagi(h)))
	j_val = stochastic_sign(prob(beta,imagj(h)))
	k_val = stochastic_sign(prob(beta,imagk(h)))
	return quaternion(r_val, i_val, j_val, k_val)
end

function internal_potential(W::AbstractArray{Quaternion{Float64}}, x::AbstractVector{Quaternion{Float64}},cue_pattern::AbstractVector{Float64},pos::Integer,s::Float64,L::Integer)
	e1 = dot(W[:,pos],x)

	if L==2
		val = x[pos]
		idx = pos*2
		e2 = quaternion(cue_pattern[idx-1] * imagi(val),
						cue_pattern[idx-1] * real(val), 
						cue_pattern[idx  ] * imagk(val),
						cue_pattern[idx  ] * imagj(val))
	elseif L==4
		val = x[pos]
		idx = pos*3
		e2 = quaternion(cue_pattern[idx-2] * imagi(val),
						cue_pattern[idx-2] * real(val), 
						cue_pattern[idx-1] * real(val),
						cue_pattern[idx  ] * real(val))
	else
		@assert(false,"unimplemented")
	end

	return (e1 + s*e2)
end

function internal_potential(W::AbstractArray{Complex{Float64}}, x::AbstractVector{Complex{Float64}},cue_pattern::AbstractVector{Float64},pos::Integer,s::Float64,L::Integer)
	e1 = dot(W[:,pos],x)

	if L==2
		e2 = cue_pattern[pos] * complex(imag(x[pos]), real(x[pos]))
	else
		@assert(false,"unimplemented")
	end

	return (e1 + s*e2)
end

function internal_potential(W::AbstractArray{Float64}, x::AbstractVector{Float64},cue_pattern::AbstractVector{Float64},pos::Integer,s::Float64,L::Integer)
	e1 = dot(W[:,pos],x)

	if L==2
		n = length(cue_pattern) 
		pos_cue = (pos -1)%n+1
		if pos > n
			e2 = cue_pattern[pos_cue] * x[pos_cue]
		else
			e2 = cue_pattern[pos_cue] * x[pos+n]
		end
	elseif L==4
		n = round(Int,length(x)/L)
		pos_cue = (pos -1)%n+1
		block = ifloor((pos-1)/n)
		@switch block begin
			0; e2 = cue_pattern[pos_cue] * x[pos_cue+1*n]
			1; e2 = cue_pattern[pos_cue] * x[pos_cue]
			2; e2 = cue_pattern[pos_cue+n] * x[pos_cue]
			3; e2 = cue_pattern[pos_cue+2*n] * x[pos_cue]
		end
	else
		@assert(false,"unimplemented")
	end

	return (e1 + s*e2)
end

function update_cue_async!(net::HopfieldNetwork, cue_pattern::AbstractVector,L::Integer, s::Real, beta::Real)
	ord = randperm(length(net.state))
	for i in ord
		h = internal_potential(net.weight,net.state,cue_pattern,i,s,L)
		net.state[i] = activatefunc(beta, h)
	end
end

function update_cue_sync!(net::HopfieldNetwork, cue_pattern::AbstractVector, L::Integer, s::Real, beta::Real)
	state = copy(net.state)
	for i = 1:length(net.state)
		h = internal_potential(net.weight,state,cue_pattern,i,s,L)
		net.state[i] = activatefunc(beta, h)
	end
end

function associate!{T1 <: Number, T2 <: Real}(net::HopfieldNetwork, init_pattern::AbstractVector{T1}, cue_pattern::AbstractVector{T2},L::Integer, s::Real, beta::Real, gamma::Real, iterations::Integer = 1_000, sync::Bool = false)
	copy!(net.state, init_pattern)
	associate!(net,cue_pattern,L,s,beta,gamma, iterations,sync)
end

function associate!{T <: Number}(net::HopfieldNetwork, cue_pattern::AbstractVector{T},L::Integer, s::Real, beta::Real, gamma::Real, iterations::Integer = 1_000, sync::Bool = false)
	if sync
		update! = update_cue_sync!
	else
		update! = update_cue_async!
	end
	beta_step = beta
	for i in 1:iterations
		update!(net, cue_pattern,L, s, beta_step)
		beta_step = beta_timestep(beta_step, gamma)
	end
	return copy(net.state)
end

function associate_step_async!{T <: Number}(net::HopfieldNetwork, cue_pattern::AbstractVector{T},L::Integer, s::Real, beta::Real, gamma::Real)
	update_cue_async!(net, cue_pattern, L, s, beta)
	return beta_timestep(beta, gamma)
end

function associate_step_sync!{T <: Number}(net::HopfieldNetwork, cue_pattern::AbstractVector{T},L::Integer, s::Real, beta::Real, gamma::Real)
	update_cue_sync!(net, cue_pattern, L, s, beta)
	return beta_timestep(beta, gamma)
end

function beta_timestep(beta::Real, gamma::Real)
	return gamma * beta;
end

