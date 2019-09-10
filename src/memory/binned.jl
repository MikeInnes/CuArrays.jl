module BinnedPool

import Base.GC: gc
import ..stats

using CUDAdrv

using TimerOutputs
import ..to


# binned memory pool allocator
#
# the core design is a pretty simple:
# - bin allocations into multiple pools according to their size (see `poolidx`)
# - when requested memory, check the pool for unused memory, or allocate dynamically
# - conversely, when released memory, put it in the appropriate pool for future use
#
# to avoid memory hogging and/or trashing the Julia GC:
# - keep track of used and available memory, in order to determine the usage of each pool
# - keep track of each pool's usage, as well as a window of previous usages
# - regularly release memory from underused pools (see `reclaim(false)`)
#
# possible improvements:
# - context management: either switch contexts when performing memory operations,
#                       or just use unified memory for all allocations.
# - per-device pools

const pool_lock = ReentrantLock()

const usage = Ref(0)
const usage_limit = Ref{Union{Nothing,Int}}(nothing)


## tunables

const MAX_POOL = 100*1024^2 # 100 MiB

const USAGE_WINDOW = 5

# min and max time between successive background task iterations.
# when the pool usages don't change, scan less regularly.
#
# together with USAGE_WINDOW, this determines how long it takes for objects to get reclaimed
const MIN_DELAY = 1.0
const MAX_DELAY = 5.0


## infrastructure

const pools_used = Vector{Set{Mem.Buffer}}()
const pools_avail = Vector{Vector{Mem.Buffer}}()

poolidx(n) = ceil(Int, log2(n))+1
poolsize(idx) = 2^(idx-1)

function create_pools(idx)
  if length(pool_usage) >= idx
    # fast-path without taking a lock
    return
  end

  lock(pool_lock) do
    while length(pool_usage) < idx
      push!(pool_usage, 1)
      push!(pool_history, initial_usage)
      push!(pools_used, Set{Mem.Buffer}())
      push!(pools_avail, Vector{Mem.Buffer}())
    end
  end
end


## timings




## management

const initial_usage = Tuple(1 for _ in 1:USAGE_WINDOW)

const pool_usage = Vector{Float64}()
const pool_history = Vector{NTuple{USAGE_WINDOW,Float64}}()

# allocation traces
const tracing = parse(Bool, get(ENV, "CUARRAYS_TRACE_POOL", "false"))
const BackTrace = Vector{Union{Ptr{Nothing}, Base.InterpreterIP}}
const alloc_sites = Dict{Mem.Buffer, Tuple{Int, BackTrace}}()
const alloc_collectables = Dict{BackTrace, Tuple{Int, Int, Int}}()

# scan every pool and manage the usage history
#
# returns a boolean indicating whether any pool is active (this can be a false negative)
function scan()
  gc(false) # quick, incremental collection

  active = false

  @inbounds for pid in 1:length(pool_history)
    nused = length(pools_used[pid])
    navail = length(pools_avail[pid])
    history = pool_history[pid]

    if nused+navail > 0
      usage = pool_usage[pid]
      current_usage = nused / (nused + navail)

      # shift the history window with the recorded usage
      history = pool_history[pid]
      pool_history[pid] = (Base.tail(pool_history[pid])..., usage)

      # reset the usage with the current one
      pool_usage[pid] = current_usage

      if usage != current_usage
        active = true
      end
    else
      pool_usage[pid] = 1
      pool_history[pid] = initial_usage
    end
  end

  active
end

# reclaim unused buffers
function reclaim(full::Bool=false, target_bytes::Int=typemax(Int))
  stats.total_time += Base.@elapsed begin
    # find inactive buffers
    @timeit to[] "scan" begin
      pools_inactive = Vector{Int}(undef, length(pools_avail)) # pid => buffers that can be freed
      if full
        # consider all currently unused buffers
        for (pid, avail) in enumerate(pools_avail)
          pools_inactive[pid] = length(avail)
        end
      else
        # only consider inactive buffers
        @inbounds for pid in 1:length(pool_usage)
          nused = length(pools_used[pid])
          navail = length(pools_avail[pid])
          recent_usage = (pool_history[pid]..., pool_usage[pid])

          if navail > 0
            # reclaim as much as the usage allows
            reclaimable = floor(Int, (1-maximum(recent_usage))*(nused+navail))
            pools_inactive[pid] = reclaimable
          else
            pools_inactive[pid] = 0
          end
        end
      end
    end

    # reclaim buffers (in reverse, to discard largest buffers first)
    @timeit to[] "reclaim" begin
      for pid in reverse(eachindex(pools_inactive))
        bytes = poolsize(pid)
        avail = pools_avail[pid]

        bufcount = pools_inactive[pid]
        @assert bufcount <= length(avail)
        for i in 1:bufcount
          buf = pop!(avail)

          stats.actual_nfree += 1
          stats.cuda_time += Base.@elapsed Mem.free(buf)
          stats.actual_free += bytes
          usage[] -= bytes

          target_bytes -= bytes
          target_bytes <= 0 && return true
        end
      end
    end
  end

  return false
end


## allocator state machine

function try_cuda_alloc(bytes)
  # check the memory allocation limit
  if usage_limit[] !== nothing
    if usage[] + bytes > usage_limit[]
      return
    end
  end

  # try the actual allocation
  try
    stats.cuda_time += Base.@elapsed begin
      buf = Mem.alloc(Mem.Device, bytes)
      usage[] += bytes
    end
    stats.actual_nalloc += 1
    stats.actual_alloc += bytes
    return buf
  catch ex
    ex == CUDAdrv.ERROR_OUT_OF_MEMORY || rethrow()
  end

  return
end

function try_alloc(bytes, pid=-1)
  # NOTE: checking the pool is really fast, and not included in the timings
  if pid != -1 && !isempty(pools_avail[pid])
    return pop!(pools_avail[pid])
  end

  @timeit to[] "1 try alloc" begin
    let buf = try_cuda_alloc(bytes)
      buf !== nothing && return buf
    end
  end

  # trace buffers that are ready to be collected by the Julia GC.
  # such objects hinder efficient memory management, and maybe should be `unsafe_free!`d
  if tracing
    alloc_sites_old = copy(alloc_sites)

    gc(true)

    for (buf, (bytes, bt)) in sort(collect(alloc_sites_old), by=x->x[2][1])
      if !haskey(alloc_sites, buf)
        if !haskey(alloc_collectables, bt)
          alloc_collectables[bt] = (1, bytes, bytes)
        else
          nalloc, _, total_bytes = alloc_collectables[bt]
          alloc_collectables[bt] = (nalloc+1, bytes, bytes+total_bytes)
        end
      end
    end
  end

  @timeit to[] "2 gc(false)" begin
    gc(false) # incremental collection
  end

  if pid != -1 && !isempty(pools_avail[pid])
    return pop!(pools_avail[pid])
  end

  # TODO: we could return a larger allocation here, but that increases memory pressure and
  #       would require proper block splitting + compaction to be any efficient.

  @timeit to[] "3 reclaim unused" begin
    reclaim(true, bytes)
  end

  @timeit to[] "4 try alloc" begin
    let buf = try_cuda_alloc(bytes)
      buf !== nothing && return buf
    end
  end

  @timeit to[] "5 gc(true)" begin
    gc(true) # full collection
  end

  if pid != -1 && !isempty(pools_avail[pid])
    return pop!(pools_avail[pid])
  end

  @timeit to[] "6 reclaim unused" begin
    reclaim(true, bytes)
  end

  @timeit to[] "7 try alloc" begin
    let buf = try_cuda_alloc(bytes)
      buf !== nothing && return buf
    end
  end

  @timeit to[] "8 reclaim everything" begin
    reclaim(true)
  end

  @timeit to[] "9 try alloc" begin
    let buf = try_cuda_alloc(bytes)
      buf !== nothing && return buf
    end
  end

  if tracing
    for (buf, (bytes, bt)) in alloc_sites
      st = stacktrace(bt, false)
      Core.print(Core.stderr, "WARNING: outstanding a GPU allocation of $(Base.format_bytes(bytes))")
      Base.show_backtrace(Core.stderr, st)
      Core.println(Core.stderr)
    end
  end

  throw(OutOfMemoryError())
end


## interface

function init(;limit=nothing)
  create_pools(30) # up to 512 MiB

  managed = parse(Bool, get(ENV, "CUARRAYS_MANAGED_POOL", "true"))
  if managed
    delay = MIN_DELAY
    @async begin
      while true
        @timeit to[] "background task" lock(pool_lock) do
          if scan()
            delay = MIN_DELAY
          else
            delay = min(delay*2, MAX_DELAY)
          end

          reclaim()
        end

        sleep(delay)
      end
    end
  end

  verbose = haskey(ENV, "CUARRAYS_MANAGED_POOL")
  if verbose
    atexit(()->begin
      Core.println("""
        Pool statistics (managed: $(managed ? "yes" : "no")):
         - requested alloc/free: $(stats.req_nalloc)/$(stats.req_nfree) ($(Base.format_bytes(stats.req_nalloc))/$(Base.format_bytes(stats.req_free)))
         - actual alloc/free: $(stats.actual_nalloc)/$(stats.actual_nfree) ($(Base.format_bytes(stats.actual_alloc))/$(Base.format_bytes(stats.actual_free)))""")
    end)
  end
end

deinit() = throw("Not implemented")

function alloc(bytes)
  # 0-byte allocations shouldn't hit the pool
  bytes == 0 && return Mem.alloc(Mem.Device, 0)

  stats.req_nalloc += 1
  stats.req_alloc += bytes
  stats.total_time += Base.@elapsed begin
    # only manage small allocations in the pool
    if bytes <= MAX_POOL
      pid = poolidx(bytes)
      create_pools(pid)
      alloc_bytes = poolsize(pid)

      @inbounds used = pools_used[pid]
      @inbounds avail = pools_avail[pid]

      lock(pool_lock) do
        buf = @timeit to[] "pooled alloc" try_alloc(alloc_bytes, pid)

        # mark the buffer as used
        push!(used, buf)

        # update pool usage
        current_usage = length(used) / (length(avail) + length(used))
        pool_usage[pid] = max(pool_usage[pid], current_usage)
      end
    else
      buf = @timeit to[] "large alloc" try_alloc(bytes)
    end
  end

  if tracing
    alloc_sites[buf] = (bytes, backtrace())
  end

  buf
end

function dealloc(buf, bytes)
  # 0-byte allocations shouldn't hit the pool
  bytes == 0 && return

  stats.req_nfree += 1
  stats.user_free += bytes
  stats.total_time += Base.@elapsed begin
    # was this a pooled buffer?
    if bytes <= MAX_POOL
      pid = poolidx(bytes)
      @assert pid <= length(pools_used)

      @inbounds used = pools_used[pid]
      @inbounds avail = pools_avail[pid]

      lock(pool_lock) do
        # mark the buffer as available
        delete!(used, buf)
        push!(avail, buf)

        # update pool usage
        current_usage = length(used) / (length(used) + length(avail))
        pool_usage[pid] = max(pool_usage[pid], current_usage)
      end
    else
      @timeit to[] "large dealloc" Mem.free(buf)
      usage[] -= bytes
    end
  end

  if tracing
    delete!(alloc_sites, buf)
  end

  return
end


## utilities

using Printf

function status()
  used_pool_buffers = 0
  used_pool_bytes = 0
  for (pid, pl) in enumerate(pools_used)
    bytes = poolsize(pid)
    used_pool_buffers += length(pl)
    used_pool_bytes += bytes * length(pl)
  end

  avail_pool_buffers = 0
  avail_pool_bytes = 0
  for (pid, pl) in enumerate(pools_avail)
    bytes = poolsize(pid)
    avail_pool_buffers += length(pl)
    avail_pool_bytes += bytes * length(pl)
  end

  pool_ratio = (used_pool_bytes + avail_pool_bytes) / used_bytes

  @printf("CuArrays.jl pool usage: %.2f%% (%s in use by %d buffers, %s idle)\n", 100*pool_ratio, Base.format_bytes(used_pool_bytes), used_pool_buffers, Base.format_bytes(avail_pool_bytes))

  return
end

function collectables()
  if !tracing
    error("Allocation tracing disabled, please start Julia and precompile CuArrays.jl with CUARRAYS_TRACE_POOL=1")
  end

  for (bt, (nalloc, bytes, total_bytes)) in sort(collect(alloc_collectables), by=x->x[2][3])
    st = stacktrace(bt, false)
    print("Eagerly collecting the following $nalloc GPU allocations of each $(Base.format_bytes(bytes)) would unblock $(Base.format_bytes(total_bytes)):")
    Base.show_backtrace(stdout, st)
    println()
  end
end

end
