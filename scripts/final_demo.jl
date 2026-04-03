using Revise,
    KdvSolver,
    Plots,
    BenchmarkTools,
    LinearAlgebra


# parameters
N = 1024
L = 200
c = 0.1
t_0 = 0
t_f = 50
n_iter = 500 * t_f

# intital waveform
u(x, t) = 3 * c * (sech(sqrt(c) / 2 * ((x - 100) - c * t)))^2
x, u_0 = dscrt(x -> u(x, 0), L, N)
_, u_f = dscrt(x -> u(x, t_f), L, N)

kvec = gen_kvec(L, N)
_ = yoshida_split(u_0, t_f, n_iter, kvec, N)
# print("Allocations and Speed: ")
# @btime yoshida_split(u_0, t_f, n_iter, kvec, N)

au_f = yoshida_split(u_0, t_f, n_iter, kvec, N)
println("Error (L2 norm): ", norm(au_f - u_f))

theme(:dark)
display(plot(x, u_0))
display(plot!(x, au_f, xlabel="x", ylabel="u(x,t)"))