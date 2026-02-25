module Novikov

export yoshida_split, dscrt, gen_kvec

include("utils.jl")

using FFTW, Plots, LinearAlgebra
# evolves linear component of KdV eqn for an initial condition f_0 to time t_f
# Assumed form of KDV:
# u_t + uu_x + u_xxx = 0
const global N = 1024
const global Ndiv2 = div(N, 2)

# starts for k = 0 even though according to 
# https://juliamath.github.io/AbstractFFTs.jl/stable/api/#Public-Interface
# this should cause weirdness
gen_kvec(L) = [(im * 2 * pi * k) / L for k = 0:Ndiv2]

# discretizes function from 0 to L without right endpoint
function dscrt(f, L)
    xvec = collect(0:N-1) * (L / N)
    return (x=xvec, y=f.(xvec))
end

function evolve_l!(u::Vector, uhat::Vector, t_f, kvec, plan, iplan)
    mul!(uhat, plan, u)
    @. uhat = uhat * exp(-kvec^3 * t_f)
    mul!(u, iplan, uhat)
    return nothing
end

function evolve_nl!(u::Vector, uhat::Vector, t, kvec, ks, plan, iplan, u_tmp)
    function f!(u_tmp, u::Vector)
        deriv!(u, u_tmp, uhat, 1, kvec, plan, iplan)
        @. u_tmp = -u * u_tmp
        return nothing
    end
    # time steps subject to change
    rk4!(f!, u, t, 1, ks, u_tmp)
    return nothing
end

function yoshida_split(u_0::Vector, t, n, kvec)
    # preallocations
    u = copy(u_0)
    uhat = zeros(ComplexF64, Ndiv2 + 1)
    plan = plan_rfft(u)
    iplan = plan_irfft(uhat, N)

    # rk4 preallocations
    ks = zeros(Float64, N, 4)
    u_tmp = similar(u)

    tstep = t / n
    tstepd2 = 0.5 * tstep

    # symplectic operator coefficients (Yoshida 1990)
    w3 = 0.784513610477560
    w2 = 0.235573213359357
    w1 = -1.17767998417887
    w0 = 1 - 2 * (w1 + w2 + w3)
    wtvec = [w3, w3, (w3 + w2), w2, (w2 + w1), w1, (w1 + w0), w0]

    for i = 1:n
        for j = 1:2:size(wtvec)[1]
            evolve_l!(u, uhat, wtvec[j] * tstepd2, kvec, plan, iplan)
            evolve_nl!(u, uhat, wtvec[j+1] * tstep, kvec, ks, plan, iplan, u_tmp)
        end
        for j = size(wtvec)[1]-1:-2:2
            evolve_l!(u, uhat, wtvec[j] * tstepd2, kvec, plan, iplan)
            evolve_nl!(u, uhat, wtvec[j-1] * tstep, kvec, ks, plan, iplan, u_tmp)
        end
        evolve_l!(u, uhat, w3 * tstepd2, kvec, plan, iplan)
    end
    return u
end

end
