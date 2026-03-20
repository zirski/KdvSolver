using Revise, Novikov, Test, Random, BenchmarkTools, LinearAlgebra

include("../src/utils.jl")
global N = 1024

@testset "deriv" begin
    f(x) = sin(x)
    df(x) = cos(x)
    L = 2 * pi

    xvec, fvec = dscrt(f, L, N)
    # does dscrt omit f(L)?
    @test fvec[1] != fvec[end]
    _, dfvec = dscrt(df, L, N)

    # preallocations
    kvec = gen_kvec(L, N)
    fhat = zeros(ComplexF64, div(N, 2) + 1)
    f_tmp = similar(fvec)

    plan = plan_rfft(fvec)
    iplan = plan_irfft(fhat, N)

    deriv!(fvec, f_tmp, fhat, kvec, plan, iplan)
    # display(plot(xvec, dfvec .- approx_soln))
    # print(maximum(abs, approx_soln .- dfvec))

    # does deriv actually work within our tolerance?
    @test isapprox(f_tmp, dfvec, atol=1e-11)

end

@testset "rk4 - 2d system" begin
    ks = zeros(Float64, 2, 4)

    @testset "logic" begin
        function g!(dx, x)
            dx[1] = 4 * x[1] - x[2]
            dx[2] = 2 * x[1] + x[2]
            return nothing
        end
        gsolved(t) = [exp(2 * t) + 2 * exp(3 * t), 2 * exp(2 * t) + 2 * exp(3 * t)]
        x_0 = [3.0, 4.0]
        x_tmp = zeros(2)
        t = 1
        rk4!(g!, x_0, x_tmp, t, 1000, ks)
        println(gsolved(t))
        @test isapprox(x_0, gsolved(t), atol=1e-9)
    end

    # @testset "convergence" begin
    #     function g!(dx, x)
    #         dx[1] = 4 * x[1] - x[2]
    #         dx[2] = 2 * x[1] + x[2]
    #         return nothing
    #     end
    #     gsolved(t) = [exp(2 * t) + 2 * exp(3 * t), 2 * exp(2 * t) + 2 * exp(3 * t)]
    #     x_0 = [3.0, 4.0]
    #     x_tmp = similar(x_0)
    #     t = 1
    #     n = 200
    #     x_e = gsolved(t)
    #     rk4!(g!, x_0, x_tmp, t, n, ks)
    #     cerr = norm(x_0 .- x_e)
    #     eoc = 0
    #     for i = 1:3
    #         x = copy(x_0)
    #         rk4!(g!, x, x_tmp, t, n, ks)
    #         nerr = norm(x .- x_e)
    #         eoc = log2(cerr / nerr)
    #         cerr = nerr
    #         println(eoc)
    #         n *= 2
    #     end
    #     @test isapprox(eoc, 4, atol=0.5)
    # end
end

@testset "accuracy - sech^2" begin
    numtests = 10
    u(x, t, c) = 3 * c * (sech(sqrt(c) / 2 * ((x - 100) - c * t)))^2

    L = 200
    t_0 = 0
    t_f = 10
    q = 1000 * t_f
    kvec = gen_kvec(L, N)

    for i = 1:numtests
        c = rand(Float64) * 1
        xs, u_0 = dscrt(x -> u(x, 0, c), L, N)
        _, u_f = dscrt(x -> u(x, t_f, c), L, N)
        au_f = yoshida_split(u_0, t_f, q, kvec, N)
        error = norm(abs.(au_f - u_f))
        @test isapprox(error, 0, atol=1e-10)
    end
end
