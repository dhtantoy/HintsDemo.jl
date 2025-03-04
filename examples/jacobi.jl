using Plots
using IterativeSolvers
using SparseArrays

normalize(x) = x / maximum(abs, x)
func_low(x) = sin(pi * x)
func_mid(x, N) = sin(N ÷ 2 * pi * x)
func_high(x, N) = sin((N - 1) * pi * x)

function main(N=100)
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
    low = func_low.(x) |> normalize
    mid = func_mid.(x, N) |> normalize
    high = func_high.(x, N) |> normalize
    uk = high + low + mid
    err_low = Float64[]
    err_mid = Float64[]
    err_high = Float64[]
    for i = 1:200
        jacobi!(uk, A, b)
        e = uk
        push!(err_low, e' * low |> abs)
        push!(err_mid, e' * mid |> abs)
        push!(err_high, e' * high |> abs)
    end
    plot(x, [low, mid, high], label=["low" "mid" "high"], xlabel="x", ylabel="u(x)", layout= (3, 1)) |> display

    plot([log10.(err_low), log10.(err_mid), log10.(err_high)], label=["low" "mid" "high"], xlabel="iteration", ylabel="log10(residual)", title= "Residual") |> display

    plot(x, uk, label="u(x)", xlabel="x", ylabel="u(x)", title="solution") |> display
end