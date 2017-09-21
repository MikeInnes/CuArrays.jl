if (
        get(ENV, "TRAVIS", "") == "true" ||
        get(ENV, "APPVEYOR", "") == "true" ||
        get(ENV, "CI", "") == "true"
    )

    Pkg.checkout("GPUArrays", "sd/abstractgpu")
    Pkg.checkout("CUDAnative", "sd/iteration")
end

# Pkg.test runs with --check_bounds=1, forcing all bounds checks.
# This is incompatible with CUDAnative (see JuliaGPU/CUDAnative.jl#98)
if Base.JLOptions().check_bounds == 1
  run(```
    $(Base.julia_cmd())
    --color=$(Base.have_color ? "yes" : "no")
    --compilecache=$(Bool(Base.JLOptions().use_compilecache) ? "yes" : "no")
    --startup-file=$(Base.JLOptions().startupfile != 2 ? "yes" : "no")
    --code-coverage=$(["none", "user", "all"][1+Base.JLOptions().code_coverage])
    $(@__FILE__)
    ```)
  exit()
end

using CuArrays
using Base.Test, GPUArrays.TestSuite
using GPUArrays: JLArray
@testset "Device: $(CUDAnative.default_device[])" begin
    TestSuite.run_tests(CuArray)
end

# The above is equal to:
# Typ = CuArray
# GPUArrays.allowslow(false)
# TestSuite.run_gpuinterface(Typ)
# TestSuite.run_base(Typ)
# TestSuite.run_blas(Typ)
# TestSuite.run_broadcasting(Typ)
# TestSuite.run_construction(Typ)
# TestSuite.run_fft(Typ)
# TestSuite.run_linalg(Typ)
# TestSuite.run_mapreduce(Typ)
# TestSuite.run_indexing(Typ)
