## statistics

mutable struct AllocStats
  # allocation requests
  req_nalloc::Int
  req_nfree::Int
  ## in bytes
  req_alloc::Int
  req_free::Int

  # actual allocations
  actual_nalloc::Int
  actual_nfree::Int
  ## in bytes
  actual_alloc::Int
  actual_free::Int

  cuda_time::Float64
  total_time::Float64
end

const alloc_stats = AllocStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

Base.copy(alloc_stats::AllocStats) =
  AllocStats((getfield(alloc_stats, field) for field in fieldnames(AllocStats))...)


## timings

using TimerOutputs
const alloc_times = Ref{TimerOutput}()

macro alloc_time(args...)
    TimerOutputs.timer_expr(__module__, false, :(CuArrays.alloc_times[]), args...)
end


## underlying allocator

const usage = Ref(0)
const usage_limit = Ref{Union{Nothing,Int}}(nothing)

function actual_alloc(bytes)
  # check the memory allocation limit
  if usage_limit[] !== nothing
    if usage[] + bytes > usage_limit[]
      return
    end
  end

  # try the actual allocation
  try
    alloc_stats.cuda_time += Base.@elapsed begin
      buf = Mem.alloc(Mem.Device, bytes)
      usage[] += bytes
    end
    alloc_stats.actual_nalloc += 1
    alloc_stats.actual_alloc += bytes
    return buf
  catch ex
    ex == CUDAdrv.ERROR_OUT_OF_MEMORY || rethrow()
  end

  return
end

function actual_free(buf)
  alloc_stats.actual_nfree += 1
  if CUDAdrv.isvalid(buf.ctx)
    alloc_stats.cuda_time += Base.@elapsed Mem.free(buf)
  end
  alloc_stats.actual_free += bytes
  usage[] -= sizeof(buf)
end


## implementations

include("memory/binned.jl")
include("memory/dummy.jl")

const allocator = if !haskey(ENV, "CUARRAYS_ALLOCATOR")
  BinnedPool
elseif ENV["CUARRAYS_ALLOCATOR"] == "binned"
  BinnedPool
elseif ENV["CUARRAYS_ALLOCATOR"] == "dummy"
  DummyAllocator
else
  error("Invalid allocator selected")
end

# allocator API
# - init(;limit=Union{Nothing,Int})
# - deinit()
# - alloc(sz)::Ptr
const alloc = allocator.alloc
# - free(::Ptr)
const free = allocator.free
# - status(used_bytes)
#   print some stats about the usage (passing the GPU memory usage for % calculations)

function __init_memory__()
  alloc_times[] = TimerOutput()

  if haskey(ENV, "CUARRAYS_MEMORY_LIMIT")
    usage_limit[] = parse(Int, ENV["CUARRAYS_MEMORY_LIMIT"])
  else
    usage_limit[] = nothing
  end

  allocator.init()
end


## utilities

using Printf

macro allocated(ex)
    quote
        let
            local f
            function f()
                b0 = alloc_stats.req_alloc
                $(esc(ex))
                alloc_stats.req_alloc - b0
            end
            f()
        end
    end
end

macro time(ex)
    quote
        local gpu_mem_stats0 = copy(alloc_stats)
        local cpu_mem_stats0 = Base.gc_num()
        local cpu_time0 = time_ns()

        local val = $(esc(ex))

        local cpu_time1 = time_ns()
        local cpu_mem_stats1 = Base.gc_num()
        local gpu_mem_stats1 = copy(alloc_stats)

        local cpu_time = (cpu_time1 - cpu_time0) / 1e9
        local gpu_gc_time = gpu_mem_stats1.total_time - gpu_mem_stats0.total_time
        local gpu_lib_time = gpu_mem_stats1.cuda_time - gpu_mem_stats0.cuda_time
        local gpu_alloc_count = gpu_mem_stats1.req_nalloc - gpu_mem_stats0.req_nalloc
        local gpu_alloc_size = gpu_mem_stats1.req_alloc - gpu_mem_stats0.req_alloc
        local cpu_mem_stats = Base.GC_Diff(cpu_mem_stats1, cpu_mem_stats0)
        local cpu_gc_time = cpu_mem_stats.total_time / 1e9
        local cpu_alloc_count = Base.gc_alloc_count(cpu_mem_stats)
        local cpu_alloc_size = cpu_mem_stats.allocd

        Printf.@printf("%10.6f seconds", cpu_time)
        for (typ, gctime, libtime, bytes, allocs) in
            (("CPU", cpu_gc_time, 0, cpu_alloc_size, cpu_alloc_count),
             ("GPU", gpu_gc_time, gpu_lib_time, gpu_alloc_size, gpu_alloc_count))
          if bytes != 0 || allocs != 0
              allocs, ma = Base.prettyprint_getunits(allocs, length(Base._cnt_units), Int64(1000))
              if ma == 1
                  Printf.@printf(" (%d%s %s allocation%s: ", allocs, Base._cnt_units[ma], typ, allocs==1 ? "" : "s")
              else
                  Printf.@printf(" (%.2f%s %s allocations: ", allocs, Base._cnt_units[ma], typ)
              end
              print(Base.format_bytes(bytes))
              if gctime > 0
                  Printf.@printf(", %.2f%% gc time", 100*gctime/cpu_time)
                if libtime > 0
                    Printf.@printf(" of which %.2f%% spent allocating", 100*libtime/gctime)
                end
              end
              print(")")
          elseif gctime > 0
              Printf.@printf(", %.2f%% %s gc time", 100*gctime/cpu_time, typ)
          end
        end
        println()

        val
    end
end

function allocator_status()
  free_bytes, total_bytes = CUDAdrv.Mem.info()
  used_bytes = total_bytes - free_bytes
  used_ratio = used_bytes / total_bytes

  @printf("Total GPU memory usage: %.2f%% (%s/%s)\n", 100*used_ratio, Base.format_bytes(used_bytes), Base.format_bytes(total_bytes))

  allocator.status(used_bytes)
end

allocator_timings() = (show(alloc_times[]; allocations=false, sortby=:name); println())
