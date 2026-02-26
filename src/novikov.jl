module Novikov

export yoshida_split, dscrt, gen_kvec

include("utils.jl")

using FFTW, Plots, LinearAlgebra
# evolves linear component of KdV eqn for an initial condition f_0 to time t_f
# Assumed form of KDV:
# u_t + uu_x + u_xxx = 0

gen_kvec(L, N) = [(im * 2 * pi * k) / L for k = 0:div(N, 2)]

# discretizes function from 0 to L without right endpoint
function dscrt(f, L, N)
    xvec = collect(0:N-1) * (L / N)
    return (x=xvec, y=f.(xvec))
end

function evolve_l!(u, uhat, t_f, kvec, plan, iplan)
    mul!(uhat, plan, u)
    @. uhat = uhat * exp(-kvec^3 * t_f)
    mul!(u, iplan, uhat)
    return nothing
end

# u_tmp is used in 2 different ways; in deriv and f! it stores the return value, whereas in rk4! it is really a temp array to
# store the arguments to f! before storing the result of f!(ks[:, n], u_tmp) in ks[:, n]
function evolve_nl!(u, u_tmp, uhat, t, kvec, ks, plan, iplan)
    function f!(du, u)
        deriv!(u, du, uhat, 1, kvec, plan, iplan)
        @. du = -u * du
        return nothing
    end
    # time steps subject to change
    rk4!(f!, u, u_tmp, t, 1, ks)
    return nothing
end

function yoshida_split(u_0, t, q, kvec, N)
    # preallocations
    Ndiv2 = div(N, 2)
    u = copy(u_0)
    uhat = Vector{ComplexF64}(undef, Ndiv2 + 1)
    plan = plan_rfft(u)
    iplan = plan_irfft(uhat, N)

    # rk4 preallocations
    ks = zeros(Float64, N, 4)
    utmp = similar(u)

    tstep = t / q
    tstepd2 = 0.5 * tstep

    # symplectic operator coefficients (Yoshida 1990)
    w3 = 0.784513610477560
    w2 = 0.235573213359357
    w1 = -1.17767998417887
    w0 = 1 - 2 * (w1 + w2 + w3)
    wtvec = (w3, w3, (w3 + w2), w2, (w2 + w1), w1, (w1 + w0), w0)
    wtlen = 8

    for i = 1:q
        for j = 1:2:wtlen
            evolve_l!(u, uhat, wtvec[j] * tstepd2, kvec, plan, iplan)
            evolve_nl!(u, utmp, uhat, wtvec[j+1] * tstep, kvec, ks, plan, iplan)
        end
        for j = wtlen-1:-2:2
            evolve_l!(u, uhat, wtvec[j] * tstepd2, kvec, plan, iplan)
            evolve_nl!(u, utmp, uhat, wtvec[j-1] * tstep, kvec, ks, plan, iplan)
        end
        evolve_l!(u, uhat, w3 * tstepd2, kvec, plan, iplan)
    end
    return u
end

end
