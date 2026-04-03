module KdvSolver

export yoshida_split, dscrt, gen_kvec

include("utils.jl")

using FFTW, Plots, LinearAlgebra
# Assumed form of KDV:
# u_t + uu_x + u_xxx = 0

# Generates vector of complex values to be applied during derivative calculations. 
gen_kvec(L, N) = [(im * 2 * pi * k) / L for k = 0:div(N, 2)]

# Discretizes function from 0 to L without right endpoint
function dscrt(f, L, N)
    xvec = collect(0:N-1) * (L / N)
    return (x=xvec, y=f.(xvec))
end

# Linear step. calculates solution linear part of KdV (u_t + u_xxx = 0) at time t_f.
function evolve_l!(u, uhat, t_f, kvec, plan, iplan)
    mul!(uhat, plan, u)
    @. uhat = uhat * exp(-kvec^3 * t_f)
    mul!(u, iplan, uhat)
    return nothing
end

# Nonlinear step. Calculates derivative via pseudospectral methods, then iterates over one RK4 step.
# u_tmp is used in 2 different ways; in deriv() and f!() it stores the return value, whereas in rk4!() it is really a temp array to
# store the arguments to f!() before storing the result of f!(ks[:, n], u_tmp) in ks[:, n].
function evolve_nl!(u, u_tmp, uhat, t, kvec, ks, plan, iplan)
    function f!(du, u)
        deriv!(u, du, uhat, kvec, plan, iplan)
        @. du = -u * du
        return nothing
    end
    # time steps subject to change
    rk4!(f!, u, u_tmp, t, 1, ks)
    return nothing
end

# Main method; splits global evolution into a combination of linear/nonlinear steps; see Yoshida (1990) for a derivation of 
# higher-order operator splitting schemes. Method repeatedly mutates a single array containing the wave's state until stopping 
# at t = t_f. All helper arrays, including scratch buffers for FFTW and the k-vector, are preallocated once and used over and over 
# again. This way we can save space and time; main loop has 0 allocations. 
function yoshida_split(u_0, t_f, q, kvec, N)

    # general preallocations
    Ndiv2 = div(N, 2)
    u = copy(u_0)
    uhat = Vector{ComplexF64}(undef, Ndiv2 + 1)
    plan = plan_rfft(u)
    iplan = plan_irfft(uhat, N)

    # rk4 preallocations
    ks = zeros(Float64, N, 4)
    u_tmp = similar(u)

    # time steps for OS
    tstep = t_f / q
    tstepd2 = 0.5 * tstep

    # symplectic operator coefficients (Yoshida 1990)
    w3 = 0.784513610477560
    w2 = 0.235573213359357
    w1 = -1.17767998417887
    w0 = 1 - 2 * (w1 + w2 + w3)
    wtvec = (w3, w3, (w3 + w2), w2, (w2 + w1), w1, (w1 + w0), w0)
    wtlen = 8

    # Main integration loop. For speed and conciseness, some consecutive linear steps are merged and are iterated over the full
    # time step. We first loop forwards, then backwards, over wtvec to save space.
    for i = 1:q
        for j = 1:2:wtlen
            evolve_l!(u, uhat, wtvec[j] * tstepd2, kvec, plan, iplan)
            evolve_nl!(u, u_tmp, uhat, wtvec[j+1] * tstep, kvec, ks, plan, iplan)
        end
        for j = wtlen-1:-2:2
            evolve_l!(u, uhat, wtvec[j] * tstepd2, kvec, plan, iplan)
            evolve_nl!(u, u_tmp, uhat, wtvec[j-1] * tstep, kvec, ks, plan, iplan)
        end
        evolve_l!(u, uhat, w3 * tstepd2, kvec, plan, iplan)
    end
    return u
end

end
