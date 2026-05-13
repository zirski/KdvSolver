using Revise,
    Novikov,
    Plots

N = 128
L = 2 * pi
t_f = 1
q = 1000 * t_f
kvec = gen_kvec(L, N)

u(x, t) = 0.1 * cos(x)
x, u_0 = dscrt(x -> u(x, 0), L, N)
_, u_f = dscrt(x -> u(x, t_f), L, N)

# au_f = integrate(u_0, t_f, q, kvec, N)