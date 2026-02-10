include("../src/lib.jl")

import .fftderiv as ftd
using FFTW

f(x, t) = sin(x) * cos(t)
L = 2 * pi
t_f = 10
f_init = ftd.dscrt(x -> f(x, 0), L)
f_final = ftd.dscrt(x -> f(x, t_f), L)
ks = ftd.kvec(L)p
# initial conditions for later use
ics = rfft(f_init.y)
println(f_init.y.size)
println(ks.size)
# computes Fourier coefficients at time t
# coefs(t) = [ics[i] * exp(-ks[i]^(3 * t)) for i = 1:ics.size[1]]

# plot(f_final.x, f_final.y)
# plot!(f_init.x, irfft(coefs(t_f), ftd.N))

