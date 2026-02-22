module novikov

export yoshida_split, dscrt, gen_kvec

include("Utils.jl")

using FFTW, Plots
# evolves linear component of KdV eqn for an initial condition f_0 to time t_f
# Assumed form of KDV:
# u_t + uu_x + u_xxx = 0
const global N = 1024
const global Ndiv2 = div(N, 2)

# starts for k = 0 even though according to 
# https://juliamath.github.io/AbstractFFTs.jl/stable/api/#Public-Interface
# this should cause weirdness
function gen_kvec(L)
    return [(im * 2 * pi * k) / L for k = 0:Ndiv2]
end

# discretizes function from 0 to L without right endpoint
function dscrt(f, L)
    xvec = collect(0:N-1) * (L / N)
    return (x=xvec, y=f.(xvec))
end

function evolve_l(f_0::Vector, t_f, kvec)
    # provides constants for computing final fourier coefficients
    ics = rfft(f_0)
    coefs = [ics[i] * exp(-kvec[i]^3 * t_f) for i = 1:Ndiv2+1]
    return irfft(coefs, N)
end

function evolve_nl(f_0::Vector, t_f, kvec)
    f(x) = -x .* deriv(x, 1, kvec)
    # time steps subject to change
    return rk4(f, f_0, t_f, 1)
end

function yoshida_split(f_0::Vector, dt, n, kvec)
    ts = dt / n
    tsd2 = 0.5 * ts
    soln = f_0

    # symplectic operator coefficients (Yoshida 1990)
    w3 = 0.784513610477560
    w2 = 0.235573213359357
    w1 = -1.17767998417887
    w0 = 1 - 2 * (w1 + w2 + w3)

    wtvec = [w3, w3, (w3 + w2), w2, (w2 + w1), w1, (w1 + w0), w0]

    for i = 1:n
        for j = 1:2:size(wtvec)[1]
            soln = evolve_l(soln, wtvec[j] * tsd2, kvec)
            soln = evolve_nl(soln, wtvec[j+1] * ts, kvec)
        end
        for j = size(wtvec)[1]-1:-2:2
            soln = evolve_l(soln, wtvec[j] * tsd2, kvec)
            soln = evolve_nl(soln, wtvec[j-1] * ts, kvec)
        end
        soln = evolve_l(soln, w3 * tsd2, kvec)

    end
    return soln
end

end
