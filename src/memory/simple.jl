module SimplePool

# linear scan into a list of free buffers

import ..@pool_timeit, ..actual_alloc, ..actual_free

using DataStructures

using CUDAdrv


## tunables

# how much larger a buf can be to fullfil an allocation request.
# lower values improve efficiency, but increase pressure on the underlying GC.
function max_oversize(sz)
    if sz <= 2^20       # 1 MiB
        # small buffers are fine no matter
        return typemax(Int)
    elseif sz <= 2^20   # 32 MiB
        return 2^20
    else
        return 2^22
    end
end


## pooling

# sorted containers need unique keys, which the size of a buffer isn't.
# mix in the buffer address to keep the key sortable, but unique.
# the size is shifted 24 bits, and as many identifier bits are mixed in,
# supporting 16777216 unique allocations of up to 1 TiB.
unique_sizeof(buf::Mem.Buffer) = (UInt64(sizeof(buf))<<24) | (UInt64(pointer(buf)) & (2<<24-1))
const UniqueIncreasingSize = Base.By(unique_sizeof)

const available = SortedSet{Mem.Buffer}(UniqueIncreasingSize)
const allocated = Set{Mem.Buffer}()

function scan(sz)
    for buf in available
        if sz <= sizeof(buf) <= max_oversize(sz)
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

    return buf
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
    if buf !== nothing
        push!(allocated, buf)
    end
    return buf
end

function free(buf)
    delete!(allocated, buf)
    pool_free(buf)
    return
end

used_memory() = isempty(allocated) ? 0 : sum(sizeof, allocated)

cached_memory() = isempty(available) ? 0 : sum(sizeof, available)

dump() = return

end
