module SimplePool

# linear scan into a list of free buffers

import ..@pool_timeit, ..actual_alloc, ..actual_free

using CUDAdrv

using DataStructures


## tunables

# how much larger a buf can be to fullfil an allocation request.
# lower values improve efficiency, but increase pressure on the underlying GC.
const MAX_OVERSIZE_RATIO = 1


## pooling

const IncreasingSize = Base.By(Base.sizeof)

const available = SortedSet{Mem.Buffer}(IncreasingSize)
const allocated = Set{Mem.Buffer}()

function scan(sz)
    for buf in available
        if sz <= sizeof(buf) <= sz*MAX_OVERSIZE_RATIO
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
        actual_free(buf)
        freed += sizeof(buf)
    end

    return freed
end

function pool_alloc(sz)
    buf = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc(false)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc(true)" GC.gc(true)
        end

        @pool_timeit "$phase.1 scan" begin
            buf = scan(sz)
        end
        buf === nothing || break

        @pool_timeit "$phase.2 alloc" begin
            buf = actual_alloc(sz)
        end
        buf === nothing || break

        @pool_timeit "$phase.3 reclaim + alloc" begin
            reclaim(sz)
            buf = actual_alloc(sz)
        end
        buf === nothing || break
    end

    if buf === nothing
        throw(OutOfMemoryError())
    else
        return buf
    end
end

function pool_free(buf)
    push!(available, buf)
end


## interface

init() = return

function deinit()
    @assert isempty(allocated) "Cannot deinitialize memory pool with outstanding allocations"

    for buf in available
        actual_free(buf)
    end
    empty!(available)

    return
end

function alloc(sz)
    buf = pool_alloc(sz)
    push!(allocated, buf)
    return buf
end

function free(buf, sz)
    delete!(allocated, buf)
    pool_free(buf)
    return
end

used_memory() = sum(sizeof, allocated)

cached_memory() = sum(sizeof, available)

end
