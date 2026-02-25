using BenchmarkTools, Novikov, Profile

include("../src/utils.jl")

N = 1024
L = 2 * pi

# Derivative function
f(x) = sin(x)
df(x) = cos(x)

xvec, fvec = dscrt(f, L)
_, dfvec = dscrt(df, L)

# preallocations
kvec = gen_kvec(L)
fhat = zeros(ComplexF64, div(N, 2) + 1)
plan = plan_rfft(fvec)
iplan = plan_irfft(fhat, N)

println("Derivative function:")
@btime deriv!($fvec, $fhat, 1, $kvec, $plan, $iplan)

# -----------------------------------------------------------
# RK4
ks = zeros(Float64, 2, 4)
n = 1
function g!(dx, x)
    dx[1] = 4 * x[1] - x[2]
    dx[2] = 2 * x[1] + x[2]
    return nothing
end
gsolved(t) = [exp(2 * t) + 2 * exp(3 * t), 2 * exp(2 * t) + 2 * exp(3 * t)]
x_0 = [3.0, 4.0]
x_tmp = similar(x_0)
x_1 = copy(x_0)
t = 1
println("RK4:")
@btime rk4!($g!, $x_0, $t, n, $ks, x_tmp)
# rk4!(g!, x_0, t, n, ks)
# @profview_allocs rk4!(g!, x_1, t, n, ks) sample_rate = 1