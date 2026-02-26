using Revise, Novikov, Plots, BenchmarkTools

# intital waveform
f(x, t) = 3 * c * (sech(sqrt(c) / 2 * ((x - 100) - c * t)))^2

c = 0.1
L = 200
N = 1024
n_iter = 1000
t_0 = 0
t_f = 1
xvec, f_0 = dscrt(x -> f(x, t_0), L, N)
_, f_f = dscrt(x -> f(x, t_f), L, N)
kvec = gen_kvec(L, N)
f_1 = copy(f_0)
soln = yoshida_split(f_0, t_f, n_iter, kvec, N)
t = @benchmark yoshida_split($f_1, $t_f, $n_iter, $kvec, $N)
# @profview_allocs yoshida_split(f_1, t_f, n_iter, kvec) sample_rate = 1
# dump(t)

# display(plot(xvec, abs.(soln - f_f)))
# display(plot(xvec, f_f))
# display(plot!(xvec, soln))