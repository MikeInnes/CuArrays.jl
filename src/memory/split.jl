module SplittingPool

# linear scan into a list of free buffers, splitting buffers along the way

import ..@pool_timeit, ..actual_alloc, ..actual_free

using CUDAdrv

using DataStructures


## tunables

# how much larger a block can be to fullfil an allocation request.
# lower values improve efficiency, but increase pressure on the underlying GC.
# this only matters when not splitting blocks.
const MAX_RELATIVE_OVERSIZE = 2

# how much larger a block should be before splitting.
const MIN_RELATIVE_REMAINDER = 1    # relative to an allocation request
const MIN_ABSOLUTE_REMAINDER = 1024

const can_split = Ref{Bool}()


##

@enum BlockState begin
    AVAILABLE
    ALLOCATED
    FREED
end

mutable struct Block
    buf::Mem.Buffer     # base allocation
    sz::Integer         # size into it
    off::Integer        # offset into it

    state::BlockState
    prev::Union{Nothing,Block}
    next::Union{Nothing,Block}

    Block(buf, sz=sizeof(buf), off=0, state=AVAILABLE, prev=nothing, next=nothing) =
        new(buf, sz, off, state, prev, next)
    Block(b::Block) = new(b.buf, b.sz, b.off, b.state, b.prev, b.next)
end

Base.sizeof(block::Block) = block.sz
Base.pointer(block::Block) = pointer(block.buf) + block.off

convert(::Type{Mem.Buffer}, block::Block) = similar(block.buf, pointer(block), sizeof(block))

# blocks are equal if the base allocation and offset is the same
Base.hash(block::Block, h::UInt) = hash(block.buf, hash(block.off, h))
Base.:(==)(a::Block, b::Block) = (a.buf == b.buf && a.off == b.off)

iswhole(block::Block) = block.prev === nothing && block.next === nothing

using Printf
function Base.show(io::IO, block::Block)
    fields = [@sprintf("%s at %p", Base.format_bytes(sizeof(block)), pointer(block))]
    push!(fields, "$(block.state)")
    block.prev !== nothing && push!(fields, @sprintf("prev=Block(%p)", pointer(block.prev)))
    block.next !== nothing && push!(fields, @sprintf("next=Block(%p)", pointer(block.next)))

    print(io, "Block(", join(fields, ", "), ")")
end

# split a block at size `sz`, returning the newly created block
function split!(block, sz)
    @assert sz < block.sz
    split = Block(block.buf, sizeof(block) - sz, block.off + sz)
    block.sz = sz

    # update links
    split.prev = block
    split.next = block.next
    if block.next !== nothing
        block.next.prev = split
    end
    block.next = split

    return split
end

# merge a sequence of blocks `blocks`
function merge!(head, blocks...)
    for block in blocks
        @assert head.next === block
        head.sz += block.sz

        # update links
        tail = block.next
        head.next = tail
        if tail !== nothing
            tail.prev = head
        end
    end

    return head
end


## pooling

const IncreasingSize = Base.By(Base.sizeof)

const available = SortedSet{Block}(IncreasingSize)
const allocated = OrderedDict{Mem.Buffer,Block}()

using Base.Threads
const pool_lock = SpinLock()   # protect against deletion from `available`

function scan(sz; max_overhead=(can_split[] ? Inf : 2))
    max_overhead = can_split[] ? Inf : MAX_RELATIVE_OVERSIZE
    lock(pool_lock) do
        for block in available
            if sz <= sizeof(block) < max_overhead
                delete!(available, block)
                return block
            end
        end
        return nothing
    end
end

# merge split blocks. happens incrementally as part of `pool_free`,
# but requires a dedicated pass in case of alloc-heavy workloads.
function compact()
    lock(pool_lock) do
        # find the unallocated head nodes
        candidates = Set{Block}()
        for block in available
            if block.state == AVAILABLE
                # get the first unallocated node in a chain
                while block.prev !== nothing && block.prev.state == AVAILABLE
                    block = block.prev
                end
                push!(candidates, block)
            end
        end

        for head in candidates
            # construct a chain of unallocated blocks
            chain = [head]
            let block = head
                while block.next !== nothing && block.next.state == AVAILABLE
                    block = block.next
                    @assert block in available
                    push!(chain, block)
                end
            end

            # compact the chain into a single block
            if length(chain) > 1
                for block in chain
                    delete!(available, block)
                end
                block = merge!(chain...)
                push!(available, block)
            end
        end
    end

    return
end

# TODO: partial reclaim?
function reclaim(sz)
    freed = 0
    candidates = []
    lock(pool_lock) do
        # mark non-split blocks
        for block in available
            if iswhole(block)
                push!(candidates, block)
            end
        end

        # free them
        for block in candidates
            actual_free(block)
            freed += sizeof(block)
            delete!(available, block)
        end
    end

    return freed
end

const ALLOC_THRESHOLD = 1*10^6  # 1 MB
const SMALL_ROUNDOFF = 5*10^5   # 0.5 MB
const LARGE_ROUNDOFF = 1*10^6   # 1 MB

function pool_alloc(sz)
    # round off the allocation size
    # FIXME: OOM
    # roundoff = sz < ALLOC_THRESHOLD ? SMALL_ROUNDOFF : LARGE_ROUNDOFF
    # alloc_sz = cld(sz, roundoff) * roundoff
    # @warn "Rounding-off alloc" sz=Base.format_bytes(sz) alloc_sz=Base.format_bytes(alloc_sz)
    alloc_sz = sz

    block = nothing
    for phase in 1:3
        if phase == 2
            @pool_timeit "$phase.0 gc(false)" GC.gc(false)
        elseif phase == 3
            @pool_timeit "$phase.0 gc(true)" GC.gc(true)
        end

        @pool_timeit "$phase.1 scan" begin
            block = scan(sz)
        end
        block === nothing || break

        @pool_timeit "$phase.2 alloc" begin
            buf = actual_alloc(alloc_sz)
            block = Block(buf)
        end
        block === nothing || break

        if can_split[]
            @pool_timeit "$phase.3 compact + scan" begin
                compact()
                block = scan(sz)
            end
            block === nothing || break
        end

        @pool_timeit "$phase.4 reclaim + alloc" begin
            reclaim(alloc_sz)
            buf = actual_alloc(alloc_sz)
            block = Block(buf)
        end
        block === nothing || break
    end

    if block === nothing
        @error "Out of memory trying to allocate $(Base.format_bytes(sz)) ($(Base.format_bytes(sum(sizeof, values(allocated)))) allocated, $(Base.format_bytes(sum(sizeof, values(available)))) fragmented)"
        println("Allocated buffers:")
        for block in values(allocated)
            println(" - ", block)
        end
        println("Available, but fragmented buffers:")
        for block in values(available)
            println(" - ", block)
        end
        throw(OutOfMemoryError())
    else
        # maybe split the block
        # TODO: creates unaligned blocks
        remainder = sizeof(block) - sz
        if can_split[] && remainder >= MIN_ABSOLUTE_REMAINDER && remainder >= sz * MIN_RELATIVE_REMAINDER
            # NOTE: we only split when there's 100% overhead or more, or else a small
            #       split-off block could keep a large unallocated chunk alive.
            split = split!(block, sz)
            push!(available, split)
        end

        block.state = ALLOCATED
        return block
    end
end

function pool_free(block)
    block.state = AVAILABLE
    push!(available, block)

    # incremental block merging
    # NOTE: requires a spinlock, because finalizer are executed in the same task as the rest
    #       of the application (i.e., a reentrant lock will not prevent us from messing up
    #       state while being held by e.g. `scan()`)
    # FIXME: OOM
    # if can_split[] && !islocked(pool_lock)
    #     lock(pool_lock) do
    #         # specialized version of `compact()`
    #         ## get the first unallocated node in a chain
    #         head = block
    #         while head.prev !== nothing && head.prev.state == AVAILABLE
    #             head = head.prev
    #         end
    #         ## construct a chain of unallocated blocks
    #         chain = [head]
    #         let block = head
    #             while block.next !== nothing && block.next.state == AVAILABLE
    #                 block = block.next
    #                 @assert block in available
    #                 push!(chain, block)
    #             end
    #         end
    #         ## compact the chain into a single block
    #         if length(chain) > 1
    #             for block in chain
    #                 delete!(available, block)
    #             end
    #             block = merge!(chain...)
    #             push!(available, block)
    #         end
    #     end
    # end
end


## interface

function init(;split=false)
    can_split[] = split
    return
end

function deinit()
    @assert isempty(allocated) "Cannot deinitialize memory pool with outstanding allocations"

    compact()

    for buf in available
        actual_free(buf)
    end
    empty!(available)

    return
end

function alloc(sz)
    block = pool_alloc(sz)
    buf = convert(Mem.Buffer, block)
    @assert !haskey(allocated, buf)
    allocated[buf] = block
    return buf
end

function free(buf)
    block = allocated[buf]
    delete!(allocated, buf)
    pool_free(block)
    return
end

used_memory() = sum(sizeof, allocated)

cached_memory() = sum(sizeof, available)

end
