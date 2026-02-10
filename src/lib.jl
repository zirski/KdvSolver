module fftderiv
export N, kvec, deriv, dscrt
using FFTW

const global N = 1024
kvec(L) = [(im * 2 * pi * k) / L for k = 0:div(N, 2)]

function dscrt(f, L)
    xvec = collect(1:N-1) * L / N
    return (x=xvec, y=f.(xvec))
end


# computes pth derivative of a function using fft technique
function deriv(f::Vector, p, kvec, N=1024)
    if div(length(f), 2) + 1 != length(kvec)
        error("kappa vector and function vector must have the same length")
    end
    return irfft(rfft(f) .* (kvec .^ p), N)
end

end

module rk4v
export rk4
# vector-valued rk4 (autonomous)
# f: vector-valued vectorized function
function rk4(f::Function, t0, x0, tf, n)
    h = (tf - t0) / n
    x = x0
    t = t0
    hdiv2 = 0.5 * h
    for i = 1:n
        k1 = f(x, t)
        k2 = f(hdiv2 .* k1 .+ x, t + hdiv2)
        k3 = f(hdiv2 .* k2 .+ x, t + hdiv2)
        k4 = f(h .* k3 .+ x, t + h)
        x = x + (h / 6) * (k1 .+ 2 .* (k2 .+ k3) .+ k4)
        t = t + h
    end
    return x
end

end
