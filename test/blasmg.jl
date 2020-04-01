using LinearAlgebra
@testset "CUBLASMB" begin

using CuArrays.CUBLASMG
using CUDAdrv

voltas    = filter(dev->occursin("V100-PCIE-32GB", name(dev)), collect(CUDAdrv.devices()))
pascals    = filter(dev->occursin("P100-PCIE", name(dev)), collect(CUDAdrv.devices()))
@show voltas, name.(voltas)
@show pascals, name.(pascals)
#voltas = voltas[1:1]
pascals = pascals[1:1]
#device!(voltas[1])
#CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), length(voltas), voltas)
#CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), length(pascals), pascals)
CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), 1, [0])
m = 8192
n = 8192
k = 8192
@testset "element type $elty" for elty in [Float32]
    alpha = convert(elty,1.1)
    beta  = convert(elty,0.0)
    @testset "Level 3" begin
        #A = rand(elty,m*k)
        #B = rand(elty,k*n)
        #C = rand(elty,m*n)
        C = zeros(Float32, m*n)
        A = fill(Float32(2.0), m*k)
        B = fill(Float32(3.0), k*n)
        @testset "gemm!" begin
            d_C = copy(C)
            #d_C = CUBLASMG.mg_gemm!('N','N',alpha,A,B,beta,d_C, devs=voltas)
            #d_C = CUBLASMG.mg_gemm!('N','N',alpha,A,B,beta,d_C, devs=pascals)
            d_C = CUBLASMG.mg_gemm!('N','N',alpha,A,(m,k),B,(k,n),beta,d_C,(m,n))
            # compare
            #h_C = (alpha*reshape(A, m, k))*reshape(B, k, n) + beta*reshape(C, m, n)
            #@test reshape(d_C, m, n) ≈ h_C
        end
    end
end # elty

end # cublasmg testset
