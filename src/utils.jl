using Base.Cartesian

cudims(a::AbstractArray) = cudims(length(a))
function cudims(n::Integer)
  threads = 256
  Base.ceil(Int, n / threads), threads
end

@inline ind2sub_(a::AbstractArray{T,0}, i) where T = ()
@inline ind2sub_(a, i) = ind2sub(a, i)

macro cuindex(A)
  quote
    A = $(esc(A))
    i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
    i > length(A) && return
    ind2sub_(A, i)
  end
end

# Concatenation
@generated function nindex(i::Int, ls::NTuple{N}) where N
  quote
    Base.@_inline_meta
    $(foldr((n, els) -> :(i ≤ ls[$n] ? ($n, i) : (i -= ls[$n]; $els)), :(-1, -1), 1:N))
  end
end

function catindex(dim, I::NTuple{N}, shapes) where N
  @inbounds x, i = nindex(I[dim], getindex.(shapes, dim))
  x, ntuple(n -> n == dim ? i : I[n], Val{N})
end

function _cat(dim, dest, xs...)
  function kernel(dim, dest, xs)
    I = @cuindex dest
    n, I′ = catindex(dim, I, size.(xs))
    @inbounds dest[I...] = xs[n][I′...]
    return
  end
  blk, thr = cudims(dest)
  @cuda (blk, thr) kernel(dim, dest, xs)
  return dest
end

function Base.cat_t(dims::Integer, T::Type, x::CuArray, xs::CuArray...)
  catdims = Base.dims2cat(dims)
  shape = Base.cat_shape(catdims, (), size.((x, xs...))...)
  dest = Base.cat_similar(x, T, shape)
  _cat(dims, dest, x, xs...)
end

Base.vcat(xs::CuArray...) = cat(1, xs...)
Base.hcat(xs::CuArray...) = cat(2, xs...)


#=
Having only sum implemented here for device arrays is a bit random, but this is
hopefully a start to have most algorithms also work on the device arrays.
Sadly, Base.sum on it's own is still a bit too complex to just work here.
# TODO Make CuDeviceArray inherit from GPUArrays.AbstractDeviceArray to have these
# function definition in one place.
=#
function Base.sum(A::CUDAnative.CuDeviceArray{T}) where T
    acc = zero(T)
    for elem in A
        acc += elem
    end
    acc
end
