using IRTools: isexpr, IR, @dynamo, postwalk
using IRTools: meta, Pipe, finish, Variable, self
using MacroTools: @forward

import Base.Broadcast.broadcasted
import Base.Broadcast.materialize
import Base.Broadcast.Broadcasted

# TODO use a WeakKeyDict
struct CUDACtx
  array_bank::IdDict{Array,CuArray}
end

CUDACtx() = CUDACtx(IdDict{Array,CuArray}())

# Display fns for debugging, remove before committing
function Base.summary(io::IO, cx::CUDACtx)
  print(io, "IR Context for CUDA ")
  summary(io, cx.array_bank)
end

function Base.show(io::IO, cx::CUDACtx)
  print(io, "IR Context for CUDA ")
  display(cx.array_bank)
end

@forward CUDACtx.array_bank Base.getindex, Base.iterate,
			Base.setindex!, Base.empty!,
			Base.length, Base.get!
			Base.first, Base.last, Base.haskey

function _resize!(a::Array, sz::NTuple{<:Any,Integer})
  ccall(:jl_array_grow_end, Cvoid, (Any, UInt), a, prod(sz))
  ptr = convert(Ptr{Csize_t},pointer_from_objref(a))
  for i = 1:length(sz)
    unsafe_store!(ptr+8*(i+2), sz[i])
  end
  return a
end

function refill!(a::Array, b::CuArray)
  _resize!(a, size(b))
  copy!(a, b)
end

function cache(cx, x::CuArray{T,N})::Array{T,N} where {T,N}
  cpu = Array{T,N}(undef, ntuple(_->0,N))
  cx[cpu] = x
  return cpu
end
cache(cx, f) = f

# TODO: BitArray and friends would like an AbstractArray construct
function get_cached(cx::CUDACtx, arr::Array{T,N})::CuArray{T,N} where {T,N}
  get!(cx, arr, CuArray(arr))
end
get_cached(cx::CUDACtx, x) = x

function (cx::CUDACtx)(::typeof(Base._mapreducedim!), f, op, args...)
  gargs = map(x -> get_cached(cx, x), args)
  Base._mapreducedim!(f, op, gargs...) |> x-> cache(cx, x)
end

macro contextual(fs...)
  ex = Expr[]
  for f in fs
    q = quote
      function (cx::CUDACtx)(::typeof($f), args...)
        gargs = map(x -> get_cached(cx, x), args)
        cache(cx, $f(gargs...))
      end
    end
    push!(ex, q)
  end

  quote
    $(ex...)
  end
end

@contextual :+ :- :* :/ sum similar materialize

function (cx::CUDACtx)(::typeof(reshape), arr, args...)
  r = reshape(get_cached(cx, arr), args...)
  cache(cx, r)
end

@dynamo function (cx::CUDACtx)(meta...)
  ir = IR(meta...)
  ir == nothing && return

  pr = Pipe(ir)
  for (v,st) in pr
    isexpr(st.expr, :call) || continue
    ex = st.expr

    pr[v] = Expr(:call, self, ex.args...)

  end
  return finish(pr)
end

"""
  Disable `CUDACtx` for a function
"""
macro noop_pass(fs...)
  ex = [:( (cx::CUDACtx)(::typeof($f), args...) = $f(args...) ) for f in fs]

  quote
    $(ex...)
  end
end

@noop_pass get_cached NNlib.check_spdf

for f in names(NNlib)
  getfield(NNlib, f) isa Function || continue
  @eval function (cx::CUDACtx)(::typeof($f), args...)
    gargs = map(x -> get_cached(cx, x), args)
    cache(cx, $f(gargs...))
  end
end

for f in names(LinearAlgebra)
  getfield(LinearAlgebra, f) isa Function || continue
  @eval function (cx::CUDACtx)(::typeof($f), args...)
    gargs = map(x -> get_cached(cx, x), args)
    cache(cx, $f(gargs...))
  end
end

"""
  Creates a `cuda` context within which we travel
  through the entire callstack to find matrix/vector
  operations and try to offload them to a GPU.

  Example:
  ```
  cuda() do
    # do something
  end
  ```
"""
function cuda(f, ctx = CUDACtx())
  out = ctx(f)
  for (x, cx) in ctx
    length(x) == length(cx) && continue
    refill!(x, cx)
  end
  empty!(ctx)
  return out
end
