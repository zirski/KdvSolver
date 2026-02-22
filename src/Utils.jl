using FFTW

const global N = 1024

# computes pth derivative of a function using fft technique
function deriv(f::Vector, p, kvec, N=1024)
    if div(length(f), 2) + 1 != length(kvec)
        error("kvec and f size disagreement; kvec has size ", length(kvec), ", while div(f, 2) + 1 has size ", div(length(f), 2) + 1, ".")
    end
    return irfft(rfft(f) .* (kvec .^ p), N)
end
# vector-valued rk4 (autonomous)
# f: vector-valued vectorized function
function rk4(f::Function, t0, x0, tf, n)
    dt = (tf - t0) / n
    x = x0
    t = t0
    hdiv2 = 0.5 * dt
    for i = 1:n
        k1 = f(x)
        k2 = f(hdiv2 .* k1 .+ x)
        k3 = f(hdiv2 .* k2 .+ x)
        k4 = f(dt .* k3 .+ x)
        x = x + (dt / 6) * (k1 .+ 2 .* (k2 .+ k3) .+ k4)
        t = t + dt
    end
    return x
end