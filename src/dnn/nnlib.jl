# interfacing with NNlib.jl

import ..CuArrays: CuVecOrMat, CuVector

import NNlib: conv!, ∇conv_filter!, ∇conv_data!, stride, dilation, flipkernel,
  maxpool!, meanpool!, ∇maxpool!, ∇meanpool!, spatial_dims, padding, kernel_size,
  softmax, softmax!, ∇softmax!, logsoftmax, logsoftmax!, ∇logsoftmax


# Softmax

const CUDNNFloat = Union{Float16,Float32,Float64}

reshape4D(x::AbstractVector) = reshape(x, 1, 1, length(x), 1)
reshape4D(x::AbstractMatrix) = reshape(x, 1, 1, size(x)...)

function softmax!(out::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxForward(reshape4D(xs), reshape4D(out))
  return out
end

function ∇softmax!(out::CuVecOrMat{T}, Δ::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxBackward(reshape4D(softmax(xs)), reshape4D(Δ), reshape4D(out))
  return out
end

function logsoftmax!(out::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxForward(reshape4D(xs), reshape4D(out), algorithm=CUDNN_SOFTMAX_LOG)
  return out
end

function ∇logsoftmax!(out::CuVecOrMat{T}, Δ::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxBackward(reshape4D(logsoftmax(xs)), reshape4D(Δ), reshape4D(out);
                       algorithm=CUDNN_SOFTMAX_LOG)
  return out
end

∇logsoftmax(Δ::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat =
  ∇logsoftmax!(similar(xs), Δ, xs)


# Convolution

function conv!(y::CuArray{T}, x::CuArray{T}, w::CuArray{T}, cdims::DenseConvDims;
               alpha=1, algo=0) where T<:CUDNNFloat
  if version() < v"6"
    all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end

  cudnnConvolutionForward(y, x, w, cdims, alpha=alpha, algo=algo)
end

# Reference for deterministic algo:
# https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardFilter
function ∇conv_filter!(dw::CuArray{T}, x::CuArray{T}, dy::CuArray{T},
                       cdims::DenseConvDims; alpha=1, algo=isdeterministic() ? 1 : 0) where T<:CUDNNFloat
  if isdeterministic()
    if algo == 0
      @warn "algorithm 0 (CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0) is not deterministic; algorithm 1 is used instead"
      algo = 1
    end
  end

  if version() < v"6"
    all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end

  cudnnConvolutionBackwardFilter(dw, x, dy, cdims, alpha=alpha, algo=algo)
end

# Reference for deterministic algo:
# https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
function ∇conv_data!(dx::CuArray{T}, dy::CuArray{T}, w::CuArray{T},
                     cdims::DenseConvDims; alpha=1, algo=isdeterministic() ? 1 : 0) where T<:CUDNNFloat
  if isdeterministic()
    if algo == 0
      @warn "algorithm 0 (CUDNN_CONVOLUTION_BWD_DATA_ALGO_0) is not deterministic; algorithm 1 is used instead"
    end
  end

  if version() < v"6"
    all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end

  cudnnConvolutionBackwardData(dx, w, dy, cdims, alpha=alpha, algo=algo)
end

∇conv_bias!(db::CuArray{T}, dy::CuArray{T}; alpha=1, beta=0) where T<:CUDNNFloat =
  cudnnConvolutionBackwardBias(db, dy, alpha=alpha, beta=beta)

# Reference for deterministic mode:
# https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnPoolingMode_t
function maxpool!(y::CuArray{T}, x::CuArray{T}, pdims::PoolDims; mode=isdeterministic() ? 3 : 0) where T<:CUDNNFloat
  if isdeterministic()
    if model != 3
      @warn "mode $mode is not deterministic; mode 3 (CUDNN_POOLING_MAX_DETERMINISTIC) is used instead"
    end
  end
  return cudnnPoolingForward(y, x, pdims; mode=mode)
end

function ∇maxpool!(dx::CuArray{T}, dy::CuArray{T}, y::CuArray{T}, x::CuArray{T},
          pdims::PoolDims, mode=isdeterministic() ? 3 : 0) where T<:CUDNNFloat
  if isdeterministic()
    if model != 3
      @warn "mode $mode is not deterministic; mode 3 (CUDNN_POOLING_MAX_DETERMINISTIC) is used instead"
    end
  end
  return cudnnPoolingBackward(dx, dy, x, y, pdims, mode=mode)
end

meanpool!(y::CuArray{T}, x::CuArray{T}, pdims::PoolDims) where T<:CUDNNFloat =
  cudnnPoolingForward(y, x, pdims, mode=1)

∇meanpool!(dx::CuArray{T}, dy::CuArray{T}, y::CuArray{T}, x::CuArray{T},
           pdims::PoolDims) where T<:CUDNNFloat =
  cudnnPoolingBackward(dx, dy, x, y, pdims, mode=1)
