module DummyPool

# dummy allocator that passes through any requests, calling into the GC if that fails.

import ..@alloc_time, ..actual_alloc, ..actual_free

init() = return

const usage = Ref(0)

function alloc(bytes)
    buf = nothing
    for phase in 1:3
        if phase == 2
            @alloc_time "$phase.0 gc(false)" GC.gc(false)
        elseif phase == 3
            @alloc_time "$phase.0 gc(true)" GC.gc(true)
        end

        @alloc_time "$phase.1 alloc" begin
            buf = actual_alloc(bytes)
        end
        buf === nothing || break
    end

    if buf === nothing
        throw(OutOfMemoryError())
    else
        usage[] += bytes
        return buf
    end
end

function free(buf, bytes)
    usage[] -= bytes
    actual_free(buf, bytes)
end

used_memory() = usage[]

cached_memory() = 0

end
