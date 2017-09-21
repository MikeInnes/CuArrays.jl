__precompile__()

module CuArrays

using CUDAdrv, CUDAnative, GPUArrays

export CuArray, CuVector, CuMatrix, cu

include("array.jl")
include("utils.jl")
include("reduction.jl")
include("blas.jl")
include("fft.jl")
include("dnn.jl")

end # module
