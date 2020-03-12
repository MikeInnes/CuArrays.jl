@testset "Statistics" begin

using CuArrays
using Statistics

@testset "std" begin
    A = rand(10, 1, 2)
    cuA = cu(A)

    @test std(cuA[:, 1, 1]) ≈ std(A[:, 1, 1])
    @test std(cuA) ≈ std(A)
    @test std(cuA, corrected=true) ≈ std(cuA, corrected=true, dims=:) ≈ std(A, corrected=true)
    @test collect(std(cuA, dims=1)) ≈ std(A, dims=1)
end

end
