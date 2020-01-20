# This file is heavlily adopted from https://github.com/JuliaMath/SpecialFunctions.jl.
# License is MIT: http://julialang.org/license

function lgamma(x)
  return CUDAnative.lgamma(x)
end

function digamma(x)
  if x <= 0 # reflection formula
    ψ = -π / CUDAnative.tan(π * x)
    x = 1 - x
  else
    ψ = zero(x)
  end
  if x < 7
    # shift using recurrence formula
    ν = one(x)
    n = 7 - CUDAnative.floor(x)
    while ν <= n - 1
      ψ -= inv(x + ν)
      ν += one(x)
    end
    ψ -= inv(x)
    x += n
  end
  t = inv(x)
  ψ += CUDAnative.log(x) - 0.5 * t
  t *= t # 1/z^2
  # the coefficients here are Float64(bernoulli[2:9] .// (2*(1:8)))
  ψ -= t * @evalpoly(t,0.08333333333333333,-0.008333333333333333,0.003968253968253968,-0.004166666666666667,0.007575757575757576,-0.021092796092796094,0.08333333333333333,-0.4432598039215686)
  return ψ
end

function _trigamma(x)
  ψ = zero(x)
  if x < 8
    # shift using recurrence formula
    n = 8 - CUDAnative.floor(x)
    ψ += inv(x)^2
    ν = one(x)
    while ν <= n - 1
      ψ += inv(x + ν)^2
      ν += one(x)
    end
    x += n
  end
  t = inv(x)
  w = t * t # 1/z^2
  ψ += t + 0.5 * w
  # the coefficients here are Float64(bernoulli[2:9])
  ψ += t * w * @evalpoly(w,0.16666666666666666,-0.03333333333333333,0.023809523809523808,-0.03333333333333333,0.07575757575757576,-0.2531135531135531,1.1666666666666667,-7.092156862745098)
  return ψ
end

function trigamma(x)
  if x <= 0 # reflection formula
    return (π / CUDAnative.sin(π * x))^2 - _trigamma(1 - x)
  else
    return _trigamma(x)
  end
end

lbeta(x, y) = lgamma(x) + lgamma(y) - lgamma(x + y)
