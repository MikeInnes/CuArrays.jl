using CuArrays, Test
using CuArrays.NNlib

# Check simple ops work and broadcast
@testset "simple ops" begin
  W = rand(5, 5)
  b = rand(5)
  op = cuda(() -> W*b)
  @test op ≈ W*b
  @test op isa Array

  a = rand(10)
  b = rand(10)

  r = cuda() do
    a + b
  end
  @test r isa Array

  r = cuda() do
    a .+ b
  end
  @test r isa Array
end

# Check that functions happen
@testset "linear" begin
  linear(x, W, b) = (x * W) .+ b
  w = rand(10, 10)
  b = zeros(10)
  x = rand(10,10)
  r = cuda() do
    linear(x, w, b)
  end
  @test r isa Array{Float32}
end

# check that NNlib is wrapped correctly
@testset "conv Context" begin
  w = rand(Float32, 3, 3, 3, 16)
  r = rand(Float32, 32, 32, 3, 1)
  g = cuda() do
    conv(r, w)
  end
  g = conv(r, w)
  @test c ≈ g
  @test g isa Array
end
