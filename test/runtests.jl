using Revise, Novikov, Test, Random, BenchmarkTools, LinearAlgebra

include("../src/utils.jl")

@testset "deriv" begin
    N = 1024
    f(x) = sin(x)
    df(x) = cos(x)
    L = 2 * pi

    xvec, fvec = dscrt(f, L)
    # does dscrt omit f(L)?
    @test fvec[1] != fvec[end]
    _, dfvec = dscrt(df, L)

    # preallocations
    kvec = gen_kvec(L)
    p = plan_rfft(fvec)
    ip = inv(p)
    fhat = zeros(ComplexF64, div(N, 2) + 1)
    f_tmp = similar(fvec)

    deriv!(fvec, f_tmp, fhat, 1, kvec, p, ip)
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
        println(x_0)
        println(gsolved(t))
        @test isapprox(x_0, gsolved(t), atol=1e-9)
    end

    @testset "convergence" begin
        function g!(dx, x)
            dx[1] = 4 * x[1] - x[2]
            dx[2] = 2 * x[1] + x[2]
            return nothing
        end
        gsolved(t) = [exp(2 * t) + 2 * exp(3 * t), 2 * exp(2 * t) + 2 * exp(3 * t)]
        x_0 = [3.0, 4.0]
        x_tmp = similar(x_0)
        t = 1
        n = 200
        x_e = gsolved(t)
        rk4!(g!, x_0, x_tmp, t, n, ks)
        cerr = norm(x_0 .- x_e)
        eoc = 0
        for i = 1:3
            x = copy(x_0)
            rk4!(g!, x, x_tmp, t, n, ks)
            nerr = norm(x .- x_e)
            eoc = log2(cerr / nerr)
            cerr = nerr
            println(eoc)
            n *= 2
        end
        @test isapprox(eoc, 4, atol=0.5)
    end
end

# @testset "rk4 - simple ODE" begin
#     g(y, lambda) = lambda * y
#     g_solved(t, lambda) = exp(lambda * t)

#     # params
#     y0 = 1
#     dt = 1
#     n = 128

#     @testset "logic" for i in 1:10
#         lambda = rand(Float64) * 5
#         y = y0
#         rk4!(x -> g(x, lambda), y, dt, n)
#         @test isapprox(g_solved(dt, lambda), y; atol=0.01)
#     end

# @testset "convergence" for i in 1:10
#     lambda = rand(Float64) * 5
#     y = y0
#     esoln = g_solved(dt, lambda)
#     n0 = 4
#     n_runs = 7
#     rk4!(x -> g(x, lambda), y, dt, n0)
#     cerror = abs(y - esoln)
#     eoc = 0
#     for i = 1:n_runs
#         n0 = n0 * 2
#         rk4!(x -> g(x, lambda), y, dt, n0)
#         nerror = abs(y - esoln)
#         eoc = log2(abs(cerror / nerror))
#         cerror = nerror
#     end
#     @test isapprox(eoc, 4; atol=0.05)
# end
# end

