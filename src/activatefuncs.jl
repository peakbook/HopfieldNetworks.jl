
export ActivateFuncs

module ActivateFuncs
import Base.sign
using QuaternionArgs
using Quaternions

export sign, quantization

# split type sign functions {{{
function sign{T<:Real}(z::Complex{T})
    return complex(sign(real(z)) ,sign(imag(z)))
end

function sign{T<:Real}(q::Quaternion{T})
    return Quaternion{T}(sign(q.q0),sign(q.q1),sign(q.q2),sign(q.q3)) 
end
# }}}

# quantization functions for multistate neurons {{{
function quantization{T<:Real}(z::Complex{T},K::Integer)
    halfphi = pi/K
    phi = 2.0*halfphi
    theta = arg(z) + halfphi
    q = floor(theta / phi)
    w = q*phi
    return complex(cos(w),sin(w))
end
function arg{T<:Real}(z::Complex{T})
    return atan2(imag(z),real(z))
end

function quantization_{T<:Real}(q::Quaternion{T},A::Integer,B::Integer,C::Integer)
    qarg = QuaternionArg(q)
    if randbool()
        phi = qarg.phi
        theta = qsign(qarg.theta, 0.5, B)
        psi = qsign(qarg.psi, 0.25, C)
    else
        phi = qsign(qarg.phi, 1.0, A)
        theta = qsign(qarg.theta, 0.5, B)
        psi = qarg.psi
    end
    return Quaternion(QuaternionArg(one(T),phi,theta,psi))
end

function quantization{T<:Real}(q::Quaternion{T},A::Integer,B::Integer,C::Integer)
    qarg = QuaternionArg(q)

    phi = qsign(qarg.phi, 1.0, A)
    theta = qsign(qarg.theta, 0.5, B)
    psi = qsign(qarg.psi, 0.25, C)

    return Quaternion(QuaternionArg(one(T),phi,theta,psi))
end

function qsign(phase::Real, coef::Real, K::Integer)
    dphase = 2*pi/K*coef
    phase0 = -pi*coef
    flg = false
    p = phase0

    # phase regulation
    phase = phase < phase0 ? phase0 : (phase >= -phase0 ? -phase0-0.5*dphase : phase)
    for i=1:K
        if p <= phase && phase < p+dphase
            flg = true
            break
        end
        p = phase0+dphase*i
    end
    @assert(flg)
 
    return p+0.5*dphase
end
# }}}

end
