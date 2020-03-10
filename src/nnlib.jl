using NNlib

# Activation functions
@cufunc σ(x::Real) = ifelse(x < -80, zero(x), one(x) / (one(x) + exp(-x)))

@cufunc softplus(x::Real) = ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))

@cufunc logσ(x::Real) = -softplus(-x)

@cufunc elu(x::Real, α = one(x)) = ifelse(x ≥ 0, x / one(x), α * (exp(x) - one(x)))

@cufunc function gelu(x::Real)
    p = oftype(x / 1, π)
    λ = oftype(x / 1, √(2 / p))
    α = oftype(x / 1, 0.044715)
    h = oftype(x / 1, 0.5)
    h * x * (one(x) + tanh(λ * (x + α * x^3)))
end

@cufunc swish(x::Real) = x * σ(x)

@cufunc lisht(x::Real) = x * tanh(x)

@cufunc function selu(x::Real)
    λ = oftype(x / 1, 1.0507009873554804934193349852946)
    α = oftype(x / 1, 1.6732632423543772848170429916717)
    λ * ifelse(x > 0, x / one(x), α * (exp(x) - one(x)))
end

@cufunc celu(x::Real, α::Real = one(x)) = ifelse(x ≥ 0, x / one(x), α * (exp(x/α) - one(x))) 

@cufunc logcosh(x::Real) = x + softplus(-2x) - log(oftype(x, 2))

@cufunc mish(x::Real) = x * tanh(softplus(x))

@cufunc tanhshrink(x::Real) = x - tanh(x)

# Batched matrix multiplication

_BATCHED_GEMM_LIST = [
    (:(CuArray{T, 3}), 'N'),
    (:(NNlib.BatchedTranspose{T, <:CuArray{T, 3}}), 'T'),
    (:(NNlib.BatchedAdjoint{T, <:CuArray{T, 3}}), 'C')
]

for (TA, transA) in _BATCHED_GEMM_LIST, (TB, transB) in _BATCHED_GEMM_LIST
    @eval function NNlib.batched_mul!(C::CuArray{T, 3}, A::$TA, B::$TB) where {T<:CUBLAS.CublasFloat}
        CuArrays.CUBLAS.gemm_strided_batched!($transA, $transB, one(T), NNlib._unbatch(A), NNlib._unbatch(B), zero(T), C)
        C
    end
end
