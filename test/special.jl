using Test
import SpecialFunctions
using Flux: Tracker
using CuArrays

n = 1000

xs_lgamma = randn(Float32, n); xs_lgamma_cu = cu(xs_lgamma)
xs_digamma = randn(Float32, n); xs_digamma_cu = cu(xs_digamma)
xs_trigamma = randn(Float32, n); xs_trigamma_cu = cu(xs_trigamma)
xs_lbeta_tuple = (randn(Float32, n), randn(Float32, n))
xs_lbeta_tuple = map(xs -> abs.(xs), xs_lbeta_tuple); xs_lbeta_cu_tuple = map(cu, xs_lbeta_tuple)

catgrads(grads) = cat(map(ta -> ta.data, grads)...; dims=1)
g∑fx(f, xs) = catgrads(Tracker.gradient(_xs -> sum(f.(_xs)), xs))
g∑fx(f, xs, ys) = catgrads(Tracker.gradient((_xs, _ys) -> sum(f.(_xs, _ys)), xs, ys))

results = Dict()
@testset "Forward evaluation" begin
    fn = :lgamma
    @testset "$fn" begin
        lgamma_val_cpu = @time SpecialFunctions.lgamma.(xs_lgamma)
        lgamma_val_gpu = @time CuArrays.lgamma.(xs_lgamma_cu)
        lgamma_val_gpu = Array(lgamma_val_gpu)
        for i = 1:n
            @test lgamma_val_cpu[i] ≈ lgamma_val_gpu[i]
        end
        results[fn] = (lgamma_val_cpu, lgamma_val_gpu)
    end

    fn = :digamma
    @testset "$fn" begin
        digamma_val_cpu = @time SpecialFunctions.digamma.(xs_digamma)
        digamma_val_gpu = @time CuArrays.digamma.(xs_digamma_cu)
        digamma_val_gpu = Array(digamma_val_gpu)
        for i = 1:n
            @test digamma_val_cpu[i] ≈ digamma_val_gpu[i]
        end
        results[fn] = (digamma_val_cpu, digamma_val_gpu)
    end

    fn = :trigamma
    @testset "$fn" begin
        trigamma_val_cpu = @time SpecialFunctions.trigamma.(xs_trigamma)
        trigamma_val_gpu = @time CuArrays.trigamma.(xs_trigamma_cu)
        trigamma_val_gpu = Array(trigamma_val_gpu)
        for i = 1:n
            @test trigamma_val_cpu[i] ≈ trigamma_val_gpu[i]
        end
        results[fn] = (trigamma_val_cpu, trigamma_val_gpu)
    end

    fn = :lbeta
    @testset "$fn" begin
        lbeta_val_cpu = @time SpecialFunctions.lbeta.(xs_lbeta_tuple...)
        lbeta_val_gpu = @time CuArrays.lbeta.(xs_lbeta_cu_tuple...)
        lbeta_val_gpu = Array(lbeta_val_gpu)
        for i = 1:n
            @test lbeta_val_cpu[i] ≈ lbeta_val_gpu[i]
        end
        results[fn] = (lbeta_val_cpu, lbeta_val_gpu)
    end

end

@testset "Gradient evaluation" begin
    fn = :lgamma
    @testset "$fn" begin
        lgamma_grad_cpu = @time g∑fx(SpecialFunctions.lgamma, xs_lgamma)
        lgamma_grad_gpu = @time g∑fx(CuArrays.lgamma, xs_lgamma_cu)
        lgamma_grad_gpu = Array(lgamma_grad_gpu)
        for i = 1:n
            @test lgamma_grad_cpu[i] ≈ lgamma_grad_gpu[i]
        end
    end

    fn = :digamma
    @testset "$fn" begin
        digamma_grad_cpu = @time g∑fx(SpecialFunctions.digamma, xs_digamma)
        digamma_grad_gpu = @time g∑fx(CuArrays.digamma, xs_digamma_cu)
        digamma_grad_gpu = Array(digamma_grad_gpu)
        for i = 1:n
            @test digamma_grad_cpu[i] ≈ digamma_grad_gpu[i]
        end
    end

    fn = :lbeta
    @testset "$fn" begin
        lbeta_grad_cpu = @time g∑fx(SpecialFunctions.lbeta, xs_lbeta_tuple...)
        lbeta_grad_gpu = @time g∑fx(CuArrays.lbeta, xs_lbeta_cu_tuple...)
        lbeta_grad_gpu = Array(lbeta_grad_gpu)
        for i = 1:n
            @test lbeta_grad_cpu[i] ≈ lbeta_grad_gpu[i]
        end
    end
end
