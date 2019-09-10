module DummyAllocator

# dummy allocator that passes through any requests, calling into the GC if that fails.

import ..CuArrays, ..alloc_stats, ..@alloc_time, ..actual_alloc, ..actual_free

init() = return

deinit() = return

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
        return buf
    end
end

free(buf, bytes) = actual_free(buf, bytes)

status(used_bytes) = return

end
