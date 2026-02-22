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

function evolve_nl(f_0::Vector, t_0, t_f, kvec)
    f(x) = -x .* deriv(x, 1, kvec)
    # time steps subject to change
    return rk4(f, t_0, f_0, t_f, 1)
end

function yoshida_split(f_0::Vector, t_0, t_f, n, kvec)
    dt = (t_f - t_0) / n
    # intermediate dt
    t = t_0
    soln = f_0
    for i = 1:n
        # linear step
        soln = evolve_l(soln, dt, kvec)
        # nonlinear step
        soln = evolve_nl(soln, t, dt, kvec)
    end
    return soln
end

end
