import GPUArrays: allowscalar, @allowscalar


## unified memory indexing

# > Simultaneous access to managed memory from the CPU and GPUs of compute capability lower
# > than 6.0 is not possible. This is because pre-Pascal GPUs lack hardware page faulting,
# > so coherence can’t be guaranteed. On these GPUs, an access from the CPU while a kernel
# > is running will cause a segmentation fault.
#
# > On Pascal and later GPUs, the CPU and the GPU can simultaneously access managed memory,
# > since they can both handle page faults; however, it is up to the application developer
# > to ensure there are no race conditions caused by simultaneous accesses.
const coherent = Ref(false)

function GPUArrays._getindex(xs::CuArray{T}, i::Integer) where T
  buf = buffer(xs)
  if isa(buf, Mem.UnifiedBuffer)
    coherent[] || CUDAdrv.synchronize()
    ptr = convert(Ptr{T}, buf)
    unsafe_load(ptr, i)
  else
    val = Array{T}(undef)
    copyto!(val, 1, xs, i, 1)
    val[]
  end
end

function GPUArrays._setindex!(xs::CuArray{T}, v::T, i::Integer) where T
  buf = buffer(xs)
  if isa(buf, Mem.UnifiedBuffer)
    coherent[] || CUDAdrv.synchronize()
    ptr = convert(Ptr{T}, buf)
    unsafe_store!(ptr, v, i)
  else
    copyto!(xs, i, T[v], 1, 1)
  end
end


## logical indexing

Base.getindex(xs::CuArray, bools::AbstractArray{Bool}) = getindex(xs, CuArray(bools))

function Base.getindex(xs::CuArray{T}, bools::CuArray{Bool}) where {T}
  bools = reshape(bools, prod(size(bools)))
  indices = @sync cumsum(bools)  # unique indices for elements that are true

  n = GPUArrays._getindex(indices, length(indices))  # number that are true
  ys = CuArray{T}(undef, n)

  if n > 0
    function kernel(ys::CuDeviceArray{T}, xs::CuDeviceArray{T}, bools, indices)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        @inbounds if i <= length(xs) && bools[i]
            b = indices[i]   # new position
            ys[b] = xs[i]
        end

        return
    end

    function configurator(kernel)
        config = launch_configuration(kernel.fun)

        threads = min(length(indices), config.threads)
        blocks = cld(length(indices), threads)

        return (threads=threads, blocks=blocks)
    end

    @cuda name="logical_getindex" config=configurator kernel(ys, xs, bools, indices)
  end

  unsafe_free!(indices)

  return ys
end


## findall

function Base.findall(bools::CuArray{Bool})
    indices = @sync cumsum(bools)

    n = GPUArrays._getindex(indices, length(indices))
    ys = CuArray{Int}(undef, n)

    if n > 0
        num_threads = min(n, 256)
        num_blocks = ceil(Int, length(indices) / num_threads)

        function kernel(ys::CuDeviceArray{Int}, bools, indices)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if i <= length(bools) && bools[i]
                b = indices[i]   # new position
                ys[b] = i
            end

            return
        end

        function configurator(kernel)
            config = launch_configuration(kernel.fun)

            threads = min(length(indices), config.threads)
            blocks = cld(length(indices), threads)

            return (threads=threads, blocks=blocks)
        end

        @cuda name="findall" config=configurator kernel(ys, bools, indices)
    end

    unsafe_free!(indices)

    return ys
end

function Base.findall(f::Function, A::CuArray)
    bools = map(f, A)
    ys = findall(bools)
    unsafe_free!(bools)
    return ys
end
