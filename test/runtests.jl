using Revise, novikov, Test, Random

include("../src/Utils.jl")

@testset "deriv" begin
    f(x) = sin(x)
    df(x) = cos(x)
    L = 2 * pi

    xvec, fvec = dscrt(f, L)
    _, dfvec = dscrt(df, L)
    kvec = gen_kvec(L)
    approx_soln = deriv(fvec, 1, kvec)
    # display(plot(xvec, dfvec .- approx_soln))
    # print(maximum(abs, approx_soln .- dfvec))

    # does deriv actually work within our tolerance?
    @test isapprox(approx_soln, dfvec, atol=1e-11)
    # does dscrt omit f(L)?
    @test fvec[1] != fvec[end]

end

@testset "rk4 - simple ODE" begin
    g(y, lambda) = lambda * y
    g_solved(t, lambda) = exp(lambda * t)

    # params
    y0 = 1
    dt = 1
    n = 128

    @testset "logic" for i in 1:10
        lambda = rand(Float64) * 10
        @test isapprox(g_solved(dt, lambda), rk4(x -> g(x, lambda), y0, dt, n); atol=0.01)
    end

    @testset "convergence" for i in 1:10
        lambda = rand(Float64) * 5
        n0 = 2
        n_runs = 10
        cerror = rk4(x -> g(x, lambda), y0, dt, n0)
        eoc = 0
        for i = 1:n_runs
            n0 = n0 * 2
            nerror = abs(rk4(x -> g(x, lambda), y0, dt, n0) - g_solved(dt, lambda))
            println(nerror)
            eoc = log2(abs(cerror / nerror))
            cerror = nerror
        end
        @test isapprox(eoc, 4; atol=0.01)
    end
end

@testset "rk4 - SHO"

