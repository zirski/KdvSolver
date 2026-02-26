using FFTW, LinearAlgebra

# computes pth derivative of a function using fft technique
# stores result in du
# u is only read; does not change
function deriv!(u, du, uhat, p, kvec, plan, iplan)
    mul!(uhat, plan, u)
    @. uhat = uhat * kvec^p
    mul!(du, iplan, uhat)
    return nothing
end
# vector-valued rk4 (autonomous)
# updates input vector x0 in-place: THROWS AWAY INITIAL CONDITION
# f: vector-valued vectorized function
function rk4!(f!::Function, u, u_tmp, t, n, ks)
    dt = t / n
    dtd2 = 0.5 * dt
    for i = 1:n
        @views f!(ks[:, 1], u)
        @views @. u_tmp = u + dtd2 * ks[:, 1]
        @views f!(ks[:, 2], u_tmp)
        @views @. u_tmp = dtd2 * ks[:, 2] + u
        @views f!(ks[:, 3], u_tmp)
        @views @. u_tmp = dt * ks[:, 3] + u
        @views f!(ks[:, 4], u_tmp)

        @views @. u = u + (dt / 6) * (ks[:, 1] + 2 * (ks[:, 2] + ks[:, 3]) + ks[:, 4])
    end
    return nothing
end

function rk4!(f::Function, u::Number, t, n)
    dt = t / n
    htstep = 0.5 * dt
    for i = 1:n
        k1 = f(u)
        k2 = f(htstep * k1 + u)
        k3 = f(htstep * k2 + u)
        k4 = f(dt * k3 + u)
        u = u + (dt / 6) * (k1 + 2 * (k2 + k3) + k4)
    end
    return nothing
end