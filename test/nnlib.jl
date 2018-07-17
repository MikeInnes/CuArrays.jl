using NNlib: conv, ∇conv_data, ∇conv_filter,
  maxpool, meanpool, ∇maxpool, ∇meanpool,
  softmax, ∇softmax, logsoftmax, ∇logsoftmax

info("Testing CuArrays/CUDNN")


@testset "NNlib" begin
  @test testf(conv, rand(10, 10, 3, 1), rand(2, 2, 3, 4))
  @test testf(∇conv_data, rand(9, 9, 4, 1), rand(10, 10, 3, 1), rand(2, 2, 3, 4))
  @test testf(∇conv_filter, rand(9, 9, 4, 1), rand(10, 10, 3, 1), rand(2, 2, 3, 4))

  @test testf(conv, rand(10, 10, 10, 3, 1), rand(2, 2, 2, 3, 4))
  @test testf(∇conv_data, rand(9, 9, 9, 4, 1), rand(10, 10, 10, 3, 1), rand(2, 2, 2, 3, 4))
  @test testf(∇conv_filter, rand(9, 9, 9, 4, 1), rand(10, 10, 10, 3, 1), rand(2, 2, 2, 3, 4))

  @test testf(x -> maxpool(x, (2,2)), rand(10, 10, 3, 1))
  @test testf(x -> meanpool(x, (2,2)), rand(10, 10, 3, 1))
  @test testf((x, dy) -> ∇maxpool(dy, maxpool(x, (2,2)), x, (2,2)), rand(10, 10, 3, 1), rand(5, 5, 3, 1))
  @test testf((x, dy) -> ∇meanpool(dy, meanpool(x, (2,2)), x, (2,2)), rand(10, 10, 3, 1), rand(5, 5, 3, 1))

  @test testf(x -> maxpool(x, (2,2,2)), rand(10, 10, 10, 3, 1))
  @test testf(x -> meanpool(x, (2,2,2)), rand(10, 10, 10, 3, 1))
  @test testf((x, dy) -> ∇maxpool(dy, maxpool(x, (2,2,2)), x, (2,2,2)), rand(10, 10, 10, 3, 1), rand(5, 5, 5, 3, 1))
  @test testf((x, dy) -> ∇meanpool(dy, meanpool(x, (2,2,2)), x, (2,2,2)), rand(10, 10, 10, 3, 1), rand(5, 5, 5, 3, 1))

  @testset "Convolution WorkSpace Size" begin
    x = cu(rand(10, 10, 3, 1));
    y = cu(rand(9, 9, 4, 1));
    w = cu(rand(2, 2, 3, 4));
    @test CuArrays.CUDNN.cudnnGetConvolutionForwardWorkspaceSize(y, x, w, algo = 1) == 492
    @test CuArrays.CUDNN.cudnnGetConvolutionBackwardFilterWorkspaceSize(w, x, w, y, algo = 1) == 2452
    @test CuArrays.CUDNN.cudnnGetConvolutionBackwardDataWorkspaceSize(x, x, w, y, algo = 1) == 2784
  end

  for dims in [(5,5), (5,)]
    @test testf(softmax, rand(dims))
    @test testf(∇softmax, rand(dims), rand(dims))
    @test testf(logsoftmax, rand(dims))
    @test testf(∇logsoftmax, rand(dims), rand(dims))
  end
end
