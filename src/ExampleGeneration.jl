function rhs_generation(n_samples, n_nodes, lambda=0.1, sigma=1.0, mean=0.0)
    gs = Gaussian(lambda, σ=sigma)
    conv = CovarianceFunction(1, gs)
    pts = range(0, stop=1, length=n_nodes)
    grf = GaussianRandomField(mean, conv, CirculantEmbedding(), pts)
    F = zeros(n_nodes, n_samples)
    for i = axes(F, 2)
        F[:, i] = sample(grf)
        m = maximum(abs, F[:, i])
        F[:, i] ./= m
    end
    return F
end

function plot_samples(A)
    n = size(A, 1)
    plot(range(0, 1, length= n), A, title="Samples", xlabel="x", ylabel="Value", legend=false, linewidth=2)
end

function fem_solve(F::Matrix)
    n_nodes = size(F, 1)
    domain = (0, 1)
    partition = (n_nodes - 1,)
    model = CartesianDiscreteModel(domain, partition)
    trian = Triangulation(model)
    bd_trian = BoundaryTriangulation(model, tags="boundary")
    dx = Measure(trian, 3)
    dσ = Measure(bd_trian, 2)
    test = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity= :H1)
    trial = TrialFESpace(test)
    f = zero(trial)
    Alu = assemble_matrix(trial, test) do u ,v 
        ∫(∇(u) ⋅ ∇(v))dx + ∫(u*v)dσ
    end |> lu
    b = zeros(n_nodes)
    sol = similar(F)

    for i = axes(F, 2)
        copy!(f.free_values, F[:, i])
        assemble_vector!(b, test) do v 
            ∫(f*v)dx
        end
        sol[:, i] = Alu\b
    end
    return sol
end
