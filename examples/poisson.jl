using Plots
using HintsDemo
using IterativeSolvers
using NeuralOperators
using Gridap 
using JLD2
using Lux

N = 32
N_samples = 10000
m = CartesianDiscreteModel((0, 1), N)
trian = Triangulation(m)
dx = Measure(trian, 3)
f(x) = 1/10 * sin(π * x[1]) + sin(1000*π*x[1])
V = TestFESpace(m, ReferenceFE(lagrangian, Float64, 1), conformity= :H1, dirichlet_tags= "boundary")
U = TrialFESpace(V, 0.)
op = AffineFEOperator(U, V) do u, v 
    ∫(∇(u)⋅∇(v))dx, ∫(f*v)dx
end
A = get_matrix(op)
b = get_vector(op)

u = A \ b
u_jacobi = jacobi(A, b; maxiter= 100)

# train the model
epochs = 5000
model = init_deeponet(N)
F = rhs_generation(N_samples, N);
U = fem_solve(F);
train_loss, test_loss, ps, st = train_deeponet!(model, (U, F), epochs; batch_size= 1000, dev= gpu_device())
deeponet = model[1]