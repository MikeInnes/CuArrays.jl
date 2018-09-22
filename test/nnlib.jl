using NNlib: conv, ∇conv_data, ∇conv_filter,
  maxpool, meanpool, ∇maxpool, ∇meanpool,
  softmax, ∇softmax, logsoftmax, ∇logsoftmax

@info("Testing CuArrays/CUDNN")


@testset "NNlib" begin
  @test testf(conv, rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4))
  @test testf(∇conv_data, rand(Float64, 9, 9, 4, 1), rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4))
  @test testf(∇conv_filter, rand(Float64, 9, 9, 4, 1), rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4))

  @test testf(conv, rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4))
  @test testf(∇conv_data, rand(Float64, 9, 9, 9, 4, 1), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4))
  @test testf(∇conv_filter, rand(Float64, 9, 9, 9, 4, 1), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4))

  @test testf(x -> maxpool(x, (2,2)), rand(Float64, 10, 10, 3, 1))
  @test testf(x -> meanpool(x, (2,2)), rand(Float64, 10, 10, 3, 1))
  @test testf((x, dy) -> ∇maxpool(dy, maxpool(x, (2,2)), x, (2,2)), rand(Float64, 10, 10, 3, 1), rand(Float64, 5, 5, 3, 1))
  @test testf((x, dy) -> ∇meanpool(dy, meanpool(x, (2,2)), x, (2,2)), rand(Float64, 10, 10, 3, 1), rand(Float64, 5, 5, 3, 1))

  @test testf(x -> maxpool(x, (2,2,2)), rand(Float64, 10, 10, 10, 3, 1))
  @test testf(x -> meanpool(x, (2,2,2)), rand(Float64, 10, 10, 10, 3, 1))
  @test testf((x, dy) -> ∇maxpool(dy, maxpool(x, (2,2,2)), x, (2,2,2)), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 5, 5, 5, 3, 1))
  @test testf((x, dy) -> ∇meanpool(dy, meanpool(x, (2,2,2)), x, (2,2,2)), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 5, 5, 5, 3, 1))

  @testset "Convolution WorkSpace Size" begin
    x = cu(rand(10, 10, 3, 1));
    y = cu(rand(9, 9, 4, 1));
    w = cu(rand(2, 2, 3, 4));
    @test CuArrays.CUDNN.cudnnGetConvolutionForwardWorkspaceSize(y, x, w, algo = 1) == 492
    @test CuArrays.CUDNN.cudnnGetConvolutionBackwardFilterWorkspaceSize(w, x, w, y, algo = 1) == 2452
    @test CuArrays.CUDNN.cudnnGetConvolutionBackwardDataWorkspaceSize(x, x, w, y, algo = 1) == 2784
  end

  for dims in [(5,5), (5,)]
    @test testf(softmax, rand(Float64, dims))
    @test testf(∇softmax, rand(Float64, dims), rand(Float64, dims))
    @test testf(logsoftmax, rand(Float64, dims))
    @test testf(∇logsoftmax, rand(Float64, dims), rand(Float64, dims))
  end
end
