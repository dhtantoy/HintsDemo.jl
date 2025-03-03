module HintsDemo

using GaussianRandomFields
using Plots
using Gridap
using Gridap.FESpaces
using LinearAlgebra
using NeuralOperators
using Lux
using Random
using Optimisers
using Zygote
using LuxCUDA
using MLUtils

export gpu_device
export cpu_device

include("ExampleGeneration.jl")
include("DeepONet.jl")

for n in names(@__MODULE__; all=true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
        @eval export $n
    end
end

end
