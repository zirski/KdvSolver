using Revise, novikov, Plots

# intital waveform
f(x, t) = 3 * c * (sech(sqrt(c) / 2 * ((x - 100) - c * t)))^2

c = 0.1
L = 200
n_iter = 1000
t_0 = 0
t_f = 1
xvec, f_0 = dscrt(x -> f(x, t_0), L)
_, f_f = dscrt(x -> f(x, t_f), L)
kvec = gen_kvec(L)
soln = yoshida_split(f_0, t_f, n_iter, kvec)
display(plot(xvec, abs.(soln - f_f)))











# exact solution which satisfies u_t = u_xx (heat eqn with D = 1)
# f(x, t) = exp(-t) * sin(x)
# L = 2 * pi
# t_f = 1
# f_init = ftd.dscrt(x -> f(x, 0), L)
# xvec = f_init.x
# IC = f_init.y
# f_final = ftd.dscrt(x -> f(x, t_f), L)
# exact_soln = f_final.y
# ks = ftd.kvec(L, true)
# initial conditions for later use
# ics = rfft(IC)
# computes Fourier coefficients at time t
# coefs(t) = [ics[i] * exp(ks[i]^3 * t) for i = 1:ftd.Ndiv2]

# fftsoln = irfft(coefs(t_f), ftd.N)


# plot(xvec, IC, label="IC")
# plot!(xvec, exact_soln, label="Exact solution at t_f")
# plot!(xvec, fftsoln, label="computed solution at t_f")

