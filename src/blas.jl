import CUBLAS
import GPUArrays: blas_module, blasbuffer
# Enable BLAS support via GPUArrays
blas_module(::CuArray) = CUBLAS
blasbuffer(A::CuArray) = CUDAdrv.CuArray(A)
