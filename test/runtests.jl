using Revise, novikov, Test

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

@testset "rk4" begin
    g(y, t) = (sin(t))^2 * y
    g_solved(t) = exp(0.5 * (t - cos(t) * sin(t)))
    y0 = 1
    @test isapprox(g_solved(1), rk4(g, 0, y0, 1, 1000))
end

