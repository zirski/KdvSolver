using Revise,
    Novikov,
    Plots,
    BenchmarkTools,
    Elliptic,
    Elliptic.Jacobi,
    LinearAlgebra


# parameters
m = 0.99
A = 5
N = 1024
t_0 = 0
t_f = 100
# must scale time step size with t_f otherwise things stop working
n_iter = 1000 * t_f

# intital waveform
b = sqrt(A / (12 * m))
# c = b^2 * (8 * m - 4)
# L = 2 * K(m) / b
L = 200
# u(x, t) = A * (cn(b * (x - c * t), m))^2
c = 0.1
u(x, t) = 3 * c * (sech(sqrt(c) / 2 * ((x - 100) - c * t)))^2
_, u_f = dscrt(x -> u(x, t_f), L, N)
kvec = gen_kvec(L, N)
# errs = Vector{Float64}(undef, 100)
# n_iter = 32
# for i = 1:20
#     local soln = yoshida_split(u_0, t_f, n_iter, kvec, N)
#     errs[i] = abs(norm(soln - u_f))
#     global n_iter = n_iter * 2
#     println("done ", i)
# end

# display(plot(errs))


soln = yoshida_split(u_0, t_f, n_iter, kvec, N)
# @profview_allocs yoshida_split(f_1, t_f, n_iter, kvec) sample_rate = 1
# dump(t)

# display(plot(xvec, abs.(soln - u_f), xlabel="N", ylabel="L2norm"))
display(plot(xvec, u_0))
display(plot!(xvec, soln, xlabel="N", ylabel="u(x,t)"))
savefig("plot2.png")
# println("Error (L2 norm): ", norm(soln - u_f))