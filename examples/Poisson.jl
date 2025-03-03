using Plots
using IterativeSolvers
using SparseArrays

normalize(x) = x / maximum(abs, x)
func_low(x) = sin(pi * x)
func_mid(x) = sin((N ÷ 2) * pi * x)
func_high(x) = sin(N * pi * x)

function main(N=300)
    B = zeros(N, N)
    for i = 1:N
        B[i, i] = 2
        if i > 1
            B[i, i-1] = -1
        end
        if i < N - 1
            B[i, i+1] = -1
        end
    end
    A = sparse(B)
    b = zeros(N)
    x = range(0, 1, length=N + 2)[2:end-1]
    low = func_low.(x)
    mid = func_mid.(x)
    high = func_high.(x)
    uk = high + low + mid
    err_low = Float64[]
    err_mid = Float64[]
    err_high = Float64[]
    for i = 1:100
        jacobi!(uk, A, b)
        e = uk
        push!(err_low, e' * low |> abs)
        push!(err_mid, e' * mid |> abs)
        push!(err_high, e' * high |> abs)
    end
    plot([low, mid, high], label=["low" "mid" "high"], xlabel="x", ylabel="u(x)", title="Mode") |> display
    plot([log10.(err_low), log10.(err_mid), log10.(err_high)], label=["low" "mid" "high"], xlabel="iteration", ylabel="log10(error)") |> display
    plot(x, uk, label="u(x)", xlabel="x", ylabel="u(x)", title="solution") |> display
end