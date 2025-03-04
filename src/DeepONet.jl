function init_deeponet(N_nodes; rng=Random.default_rng())
    deeponet = DeepONet(
        Chain(Dense(N_nodes => 80, relu), BatchNorm(80, relu)),
        Chain(Dense(1 => 80, tanh), Dense(80 => 80))
    )
    ps, st = Lux.setup(rng, deeponet)

    return deeponet, ps, st
end

function train_deeponet!((deeponet, ps, st), (U, F), epochs::Integer=10;
    batch_size::Integer=10,
    dev=cpu_device(),
    learning_rate::Real=5e-4)

    train_ratio = 1 - batch_size / size(U, 2)
    cdev = cpu_device()
    ps = ps |> dev 
    st = st |> dev
    x = reshape(range(0, 1, length=size(U, 1)), 1, :, 1) |> collect |> dev
    U_train, U_test = splitobs(U, at=train_ratio) |> dev
    F_train, F_test = splitobs(F, at=train_ratio) |> dev
    UF_train_batch = DataLoader((U_train, F_train), batchsize=batch_size, shuffle=false, partial=false) |> dev
    UF_test_batch = DataLoader((U_test, F_test), batchsize=batch_size, shuffle=false, partial=false) |> dev

    train_loss = Float32[]
    test_loss = Float32[]
    loss = 0.0
    loss_func = MSELoss()

    tstate = Training.TrainState(deeponet, ps, st, Adam(learning_rate))
    for i in 1:epochs, (u_test, f_test) in UF_test_batch
        for (u_train, f_train) in UF_train_batch
            _, loss, _, tstate = Training.single_train_step!(AutoZygote(), loss_func, ((f_train, x), u_train),
                tstate)
        end
        v = loss_func(deeponet((f_test, x), ps, st) |> first, u_test)
        if i % 50 == 0
            println("Epoch: $i, Loss: $loss, Test Loss: $v")
        end
        push!(train_loss, loss)
        push!(test_loss, v)
    end
    return (train_loss, test_loss, ps, st) |> cdev
end