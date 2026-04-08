using Revise,
    KdvSolver,
    Plots,
    BenchmarkTools,
    Elliptic,
    Elliptic.Jacobi,
    LinearAlgebra,
    Statistics,
    Printf

# parameters
N = 1024
t_0 = 0
t_f = 1
n_iter = 25000 * t_f

# u(x, t) = A * (cn(b * (x - c * t), m))^2

num_tests = 5

amps = rand(num_tests) * 5
params = rand(num_tests) .* (1.0 - 2 * eps()) .+ eps()
errs_mse = zeros(num_tests)
errs_l2 = zeros(num_tests)

for i = 1:num_tests
    # intital waveform function
    local b = sqrt(amps[i] / (12 * params[i]))
    local c = b^2 * (8 * params[i] - 4)
    local L = 2 * K(params[i]) / b
    local u(x, t) = amps[i] * (cn(b * (x - c * t), params[i]))^2

    local x, u_0 = dscrt(x -> u(x, t_0), L, N)
    local _, u_f = dscrt(x -> u(x, t_f), L, N)

    local kvec = gen_kvec(L, N)
    local au_f = yoshida_split(u_0, t_f, n_iter, kvec, N)

    println("(", amps[i], ", ", params[i], ")")
    err_raw = abs.(u_f - au_f)
    println(maximum(err_raw))

    errs_mse[i] = mean(abs2, err_raw)
    errs_l2[i] = sqrt(L / N) * norm(err_raw)
end

println("Maximum mean squared error: ", maximum(errs_mse))
println("Maximum L2 error: ", maximum(errs_l2), "\n")

println("EOC:")

m = 0.99
A = 10
b = sqrt(A / (12 * m))
c = b^2 * (8 * m - 4)
L = 2 * K(m) / b
u(x, t) = A * (cn(b * (x - c * t), m))^2

num_tests = 16
qs = zeros(num_tests)
qs[1] = 2
for i = 2:num_tests
    qs[i] = qs[i-1] * 2
end

errs_l2 = zeros(num_tests)
for i = 1:num_tests
    local x, u_0 = dscrt(x -> u(x, t_0), L, N)
    local _, u_f = dscrt(x -> u(x, t_f), L, N)
    local kvec = gen_kvec(L, N)
    local au_f = yoshida_split(u_0, t_f, qs[i], kvec, N)
    display(plot(x, au_f))
    err_raw = abs.(au_f - u_f)
    errs_l2[i] = sqrt(L / N) * norm(err_raw)
    if i > 1
        println(log2(errs_l2[i-1] / errs_l2[i]))
    end
end



