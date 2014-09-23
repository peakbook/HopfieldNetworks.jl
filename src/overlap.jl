#overlap funcs {{{
## for binary states {{{
export overlap, overlapm

function overlap{T<:Real}(x::AbstractVector{T},y::AbstractVector{T})
    return dot(x, y)/length(x)
end

function overlap{T<:Real}(x::AbstractVector{Complex{T}},y::AbstractVector{Complex{T}})
    rsum = dot(real(x), real(y))
    isum = dot(imag(x), imag(y))
    return (rsum+isum)/length(x)*0.5
end

function overlap{T<:Real}(x::AbstractVector{Quaternion{T}},y::AbstractVector{Quaternion{T}})
    rsum = dot(real(x), real(y))
    isum = dot(imagi(x), imagi(y))
    jsum = dot(imagj(x), imagj(y))
    ksum = dot(imagk(x), imagk(y))
    return (rsum+isum+jsum+ksum)/length(x)*0.25
end
## }}}

## for multi states {{{
function overlapm{T<:Real}(x::AbstractVector{T},y::AbstractVector{T})
    return sum(x .== y)/length(x)
end

function overlapm{T<:Real}(x::AbstractVector{Complex{T}},y::AbstractVector{Complex{T}})
    rsum = sum(real(x).== real(y))
    isum = sum(imag(x).== imag(y))
    return (rsum+isum)/length(x)*0.5
end

function overlapm{T<:Real}(x::AbstractVector{Quaternion{T}},y::AbstractVector{Quaternion{T}})
    rsum = sum(real(x).== real(y))
    isum = sum(imagi(x).== imagi(y))
    jsum = sum(imagj(x).== imagj(y))
    ksum = sum(imagk(x).== imagk(y))
    return (rsum+isum+jsum+ksum)/length(x)*0.25
end
## }}}
