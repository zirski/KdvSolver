using Revise, Novikov, Plots

# intital waveform
f(x, t) = 3 * c * (sech(sqrt(c) / 2 * ((x - 100) - c * t)))^2

c = 0.01
L = 200
n_iter = 1000
t_0 = 0
t_f = 1
xvec, f_0 = dscrt(x -> f(x, t_0), L)
_, f_f = dscrt(x -> f(x, t_f), L)
kvec = gen_kvec(L)
f_1 = copy(f_0)
soln = yoshida_split(f_0, t_f, n_iter, kvec)
# @btime yoshida_split($f_1, $t_f, $n_iter, $kvec)
# @profview yoshida_split(f_1, t_f, n_iter, kvec)

display(plot(xvec, abs.(soln - f_f)))
# display(plot(xvec, f_f))
# display(plot!(xvec, soln))