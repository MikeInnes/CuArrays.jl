module SimplePool

# linear scan into a list of free buffers

import Base.GC: gc
import ..CuArrays, ..alloc_stats, ..@alloc_time, ..actual_alloc, ..actual_free

using CUDAdrv

using Printf

using DataStructures


## tunables

# how much larger a buf can be to fullfil an allocation request.
# lower values improve efficiency, but increase pressure on the underlying GC.
const MAX_OVERSIZE_RATIO = 2


## pooling

const IncreasingSize = Base.By(Base.sizeof)

const available = SortedSet{Mem.Buffer}(IncreasingSize)
const allocated = Set{Mem.Buffer}()

function scan(sz)
    for buf in available
        if sizeof(buf) >= sz && sizeof(buf) < sz * MAX_OVERSIZE_RATIO
            delete!(available, buf)
            return buf
        end
    end
    return
end

function reclaim(sz)
    freed = 0
    while freed < sz && !isempty(available)
        buf = pop!(available)
        actual_free(buf, sizeof(buf))
        freed += sizeof(buf)
    end

    return freed
end

function pool_alloc(sz)
    buf = nothing
    for phase in 1:3
        if phase == 2
            @alloc_time "$phase.0 gc(false)" GC.gc(false)
        elseif phase == 3
            @alloc_time "$phase.0 gc(true)" GC.gc(true)
        end

        @alloc_time "$phase.1 scan" begin
            buf = scan(sz)
        end
        buf === nothing || break

        @alloc_time "$phase.2 alloc" begin
            buf = actual_alloc(sz)
        end
        buf === nothing || break

        @alloc_time "$phase.3 reclaim + alloc" begin
            reclaim(sz)
            buf = actual_alloc(sz)
        end
        buf === nothing || break
    end

    if buf === nothing
        throw(OutOfMemoryError())
    else
        push!(allocated, buf)
        return buf
    end
end

function pool_free(buf)
    delete!(allocated, buf)
    push!(available, buf)
end


## interface

init() = return

function deinit()
    @assert isempty(allocated) "Cannot deinitialize memory allocator with outstanding allocations"

    for buf in available
        actual_free(buf, sizeof(buf))
    end
    empty!(available)

    return
end

function alloc(sz)
    alloc_stats.req_nalloc += 1
    alloc_stats.req_alloc += sz
    alloc_stats.total_time += Base.@elapsed begin
        @alloc_time "pooled alloc" buf = pool_alloc(sz)
    end

    return buf
end

function free(buf, sz)
    alloc_stats.req_nfree += 1
    alloc_stats.req_free += sz
    alloc_stats.total_time += Base.@elapsed begin
        @assert sizeof(buf) >= sz
        @alloc_time "pooled free" pool_free(buf)
    end

    return
end

function status(used_bytes)
  used_pool_buffers = length(allocated)
  used_pool_bytes = sum(sizeof, allocated)

  avail_pool_buffers = length(available)
  avail_pool_bytes = sum(sizeof, available)

  pool_ratio = (used_pool_bytes + avail_pool_bytes) / used_bytes

  @printf("CuArrays.jl simple pool usage: %.2f%% (%s in use by %d buffers, %s idle)\n", 100*pool_ratio, Base.format_bytes(used_pool_bytes), used_pool_buffers, Base.format_bytes(avail_pool_bytes))

  return
end

end
