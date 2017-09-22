using CUDAdrv: OwnedPtr
using CUDAnative: DevicePtr

import Base: pointer, similar, size, copy!, convert

import GPUArrays: GPUArray, unsafe_reinterpret, LocalMemory, gpu_sub2ind, _gpu_call
import GPUArrays: is_gpu, name, threads, blocks, global_memory, local_memory, device
import GPUArrays: synchronize_threads, AbstractDeviceArray

using GPUArrays: thread_blocks_heuristic

mutable struct CuArray{T,N} <: GPUArray{T,N}
  ptr::OwnedPtr{T}
  dims::NTuple{N,Int}
  function CuArray{T,N}(ptr::OwnedPtr{T}, dims::NTuple{N,Integer}) where {T,N}
    xs = new{T,N}(ptr, dims)
    Mem.retain(ptr)
    finalizer(xs, unsafe_free!)
    return xs
  end
end
size(A::CuArray) = A.dims
pointer(A::CuArray) = A.ptr

CuVector{T} = CuArray{T,1}
CuMatrix{T} = CuArray{T,2}

function unsafe_free!(xs::CuArray)
    Mem.release(xs.ptr) && CUDAdrv.isvalid(xs.ptr.ctx) && Mem.free(xs.ptr)
    return
end


function (::Type{CuArray{T,N}})(size::NTuple{N,Integer}) where {T,N}
    CuArray{T,N}(Mem.alloc(T, prod(size)), size)
end

similar(::Type{<: CuArray}, ::Type{T}, size::Base.Dims{N}) where {T, N} =
  CuArray{T,N}(size)

Base.sizeof(x::CuArray) = Base.elsize(x) * length(x)


function unsafe_reinterpret(::Type{T}, A::CuArray{ET}, size::NTuple{N, Integer}) where {T, ET, N}
    ptr = pointer(A)
    #Mem.retain(ptr) # TODO do we need to retain in cuda?
    CuArray{T, N}(OwnedPtr{T}(ptr), size)
end


function Base.copy!{T}(
        dest::Array{T}, d_offset::Integer,
        source::CuArray{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    d_offset = d_offset
    s_offset = s_offset - 1
    device_ptr = pointer(source)
    sptr = device_ptr + (sizeof(T) * s_offset)
    CUDAdrv.Mem.download(Ref(dest, d_offset), sptr, sizeof(T) * (amount))
    dest
end
function Base.copy!{T}(
        dest::CuArray{T}, d_offset::Integer,
        source::Array{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    d_offset = d_offset - 1
    s_offset = s_offset
    d_ptr = pointer(dest)
    sptr = d_ptr + (sizeof(T) * d_offset)
    CUDAdrv.Mem.upload(sptr, Ref(source, s_offset), sizeof(T) * (amount))
    dest
end


function Base.copy!{T}(
        dest::CuArray{T}, d_offset::Integer,
        source::CuArray{T}, s_offset::Integer, amount::Integer
    )
    d_offset = d_offset - 1
    s_offset = s_offset - 1
    d_ptr = pointer(dest)
    s_ptr = pointer(source)
    dptr = d_ptr + (sizeof(T) * d_offset)
    sptr = s_ptr + (sizeof(T) * s_offset)
    CUDAdrv.Mem.transfer(sptr, dptr, sizeof(T) * (amount))
    dest
end

Base.collect(x::CuArray{T,N}) where {T,N} =
  copy!(Array{T,N}(size(x)), x)


# Interop with CUDAdrv native array
# Interop with CUDAdrv native array

convert(::Type{CUDAdrv.CuArray{T,N}}, xs::CuArray{T,N}) where {T,N} =
  CUDAdrv.CuArray{T,N}(xs.dims, xs.ptr)

convert(::Type{CUDAdrv.CuArray}, xs::CuArray{T,N}) where {T,N} =
  convert(CUDAdrv.CuArray{T,N}, xs)

convert(::Type{CuArray{T,N}}, xs::CUDAdrv.CuArray{T,N}) where {T,N} =
  CuArray{T,N}(xs.ptr, xs.shape)

convert(::Type{CuArray}, xs::CUDAdrv.CuArray{T,N}) where {T,N} =
  convert(CuArray{T,N}, xs)

# Interop with CUDAnative device array

function convert(::Type{CuDeviceArray{T,N,AS.Global}}, a::CuArray{T,N}) where {T,N}
    ptr = Base.unsafe_convert(Ptr{T}, pointer(a))
    CuDeviceArray{T,N,AS.Global}(a.dims, DevicePtr{T,AS.Global}(ptr))
end

using Base: RefValue
CUDAnative.cudaconvert(a::CuArray{T,N}) where {T,N} = convert(CuDeviceArray{T,N,AS.Global}, a)
CUDAnative.cudaconvert(a::RefValue{CuArray{T,N}}) where {T,N} = RefValue(convert(CuDeviceArray{T,N,AS.Global}, a[]))

# Utils

cu(x) = x
cu(x::CuArray) = x

cu(xs::AbstractArray) = isbits(xs) ? xs : CuArray(xs)

Base.getindex(::typeof(cu), xs...) = CuArray([xs...])

#Abstract GPU interface
immutable CUKernelState end

@inline function LocalMemory(::CUKernelState, ::Type{T}, ::Val{N}, ::Val{C}) where {T, N, C}
    CUDAnative.generate_static_shmem(Val{C}, T, Val{N})
end

(::Type{AbstractDeviceArray})(A::CuDeviceArray, shape) = CuDeviceArray(shape, pointer(A))


@inline synchronize_threads(::CUKernelState) = CUDAnative.sync_threads()

for (i, sym) in enumerate((:x, :y, :z))
    for (f, fcu) in (
            (:blockidx, :blockIdx),
            (:blockdim, :blockDim),
            (:threadidx, :threadIdx),
            (:griddim, :gridDim)
        )
        fname = Symbol(string(f, '_', sym))
        cufun = Symbol(string(fcu, '_', sym))
        @eval GPUArrays.$fname(::CUKernelState)::Cuint = CUDAnative.$cufun()
    end
end

devices() = CUDAdrv.devices()
GPUArrays.device(A::CuArray) = CUDAnative.default_device[]
is_gpu(dev::CUDAdrv.CuDevice) = true
name(dev::CUDAdrv.CuDevice) = string("CU ", CUDAdrv.name(dev))
threads(dev::CUDAdrv.CuDevice) = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)

function blocks(dev::CUDAdrv.CuDevice)
    (
        CUDAdrv.attribute(dev, CUDAdrv.MAX_BLOCK_DIM_X),
        CUDAdrv.attribute(dev, CUDAdrv.MAX_BLOCK_DIM_Y),
        CUDAdrv.attribute(dev, CUDAdrv.MAX_BLOCK_DIM_Z),
    )
end

free_global_memory(dev::CUDAdrv.CuDevice) = CUDAdrv.Mem.info()[1]
global_memory(dev::CUDAdrv.CuDevice) = CUDAdrv.totalmem(dev)
local_memory(dev::CUDAdrv.CuDevice) = CUDAdrv.attribute(dev, CUDAdrv.TOTAL_CONSTANT_MEMORY)


function _gpu_call(f, A::CuArray, args::Tuple, blocks_threads::Tuple{T, T}) where T <: NTuple{N, Integer} where N
    blocks, threads = blocks_threads
    @cuda (blocks, threads) f(CUKernelState(), args...)
end
