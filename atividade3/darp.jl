using JuMP
using CPLEX
using DataFrames

println("Packages imported.")
model = Model(CPLEX.Optimizer)

#conjunto de nós (vértices) do grafo direcionado completo G(V, A) que define oproblema
# V = [
#     [-1.198, 5.573], [-6.614, -7.374], [-9.251,
#     -6.498], [0.861, 3.904], [7.976, -2.61], [4.487,
#     8.938], [-4.172, 7.835], [2.792, 5.212], [6.687,
#     -2.192], [-1.061, 6.883], [5.586, -9.865], [-9.8,
#     1.271], [4.404, 0.673], [7.032, -0.694], [3.763, 6.634],
#     [-9.45, -8.819],
# ]

V = [
    [-1.198, 6.687],
    [5.573,-2.192 ],
    [-6.614, -1.061],
    [-7.374, 6.883],
    [-9.251, 5.586],
    [-6.498, -9.865],
    [0.861,-9.8 ],
    [3.904,1.271 ],
    [7.976, 4.404],
    [-2.61, 0.673,],
    [4.487,7.032,],
    [8.938,-0.694,],
    [-4.172,3.763,],
    [7.835,6.634,],
    [2.792,-9.45,],
    [5.212, -8.819],
]

# conjunto de nós de coleta
C = Set([
    -1.198, 5.573, -6.614, -7.374, -9.251,
    -6.498, 0.861, 3.904, 7.976, -2.61, 4.487,
    8.938, -4.172, 7.835, 2.792, 5.212,
])

# conjunto de nós de entrega
E = Set([
    6.687, -2.192, -1.061, 6.883, 5.586,
    -9.865, -9.8, 1.271, 4.404, 0.673, 7.032,
    -0.694, 3.763, 6.634, -9.45, -8.819,
])

# conjunto de pedidos, cada qual relacionado a um nó de coleta e um de entrega
P = Set([
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16,
])
K = Set([1, 2]) # conjunto de veículos disponíveis para roteamento

# Parâmetros
INF = 99999999999999

n = 16 # numero de pedidos a serem servidos

# tempo de serviço (embarque ou desembarque) no nó vi ∈ C ∪ E
s = [
    3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
]

# tempo de viagem de transpor o arco (i, j)
t = [
    3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
]
# c = [] # custo de transporte do veículo k ao passar pelo arco (i, j)

# carregamento do nó vi ∈ C ∪E. É > 0 se vi ∈ C e negativo se vi ∈ C. Além disso, qi = −qn+i
q = [
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
]

Qk = 3 # capacidade do veículo k ∈ K
Tk = 480 # tempo max total de rota para cada k ∈ K

# e = [0, 0, 0, 0, 0, 0, 0, 0, 276, 32, 115, 14, 198, 160, 180, 366] # limite inferior da janela de tempo do nó vi ∈ C ∪ E
# limite inferior da janela de tempo do nó vi ∈ C ∪ E
e = [
    402, 322, 179, 138, 82, 49, 400, 298,
    276, 32, 115, 14, 198, 160, 180, 366,
]

# l = [1440, 1440, 1440, 1440, 1440, 1440, 1440, 1440, 291, 47, 130, 29, 213, 175, 195, 381] # limite superior da janela de tempo do nó vi ∈ C ∪ E
# limite superior da janela de tempo do nó vi ∈ C ∪ E
l = [
    1440 - 417,
    1440 - 337,
    1440 - 194,
    1440 - 153,
    1440 - 97,
    1440 - 64,
    1440 - 415,
    1440 - 313,
    1440 - 291,
    1440 - 47,
    1440 - 130,
    1440 - 29,
    1440 - 213,
    1440 - 175,
    1440 - 195,
    1440 - 381,
]

# tempo de viagem máxima aceito pelo usuário do pedido i
L = 30

# Variáveis
@variable(model, B[1:n] >= 0, Int) # instante em que o veículo k inicia o serviço no nó i.
@variable(model, Lk[1:n] >= 0, Int) # tempo de viagem do usuário i quando no veículo k
@variable(model, Qijk[1:n] >= 0, Int) # ocupação do veículo k quando no nó i

# variável de decisão do problema, 1 quando o arco (i, j) é percorrido pelo veículo k e 0 caso contrário.
@variable(model, x[1:n, 1:n] >= 0, Int) # conferir
@variable(model, c[1:n, 1:n] >= 0, Int) # conferir

# K = [1, 2]
# V [[1, 2]]
@objective(model, Min,
    sum(
        sum(
            sum(
                c[i, j] * x[i, j] for j in 1:length(V)
            ) for i in k:length(V) # POSSIVEL BUG
        ) for k in 1:length(K)
    )
)

print("Objective")

@constraint(model, c1[i = 1:length(C)],
    sum(
        sum(
            x[i, j] for j in 1:length(V)
        ) for k in K
    ) == 1
)

@constraint(model, c2[i = 1:length(C), k = 1:length(K)],
    sum(x[i, j] for j in 1:length(C)) - sum(x[i, j] for j in 1:length(E)) == 0
)

@constraint(model, c3[k = 1:length(K)],
    sum(
        x[1, j] for j in 1:length(V)
    ) == 1
)

@constraint(model, c4[i = 1:length(V), k = 1:length(K)],
    sum(x[j, i] for j in 1:length(V)) - sum(x[i, j] for j in 1:length(V)) == 0
)

@constraint(model, c5[k = 1:length(K)],
    sum(
        x[i, 16] for i in 1:length(V)
    ) == 1
)

@constraint(model, c6[i = 1:length(V), j = 1:length(V), k = 1:length(K)],
    B[j] >= (B[i] + s[i] + t[i]) * x[i, j]
)

@constraint(model, c7[i = 1:length(P), k = 1:length(K)],
    Lk[i] == B[i] - (B[i] + s[i]) # HACK: changed B[n+i]
)

@constraint(model, c8[k = 1:length(K)],
    B[16] - B[1] <= Tk # HACK: B[2*n + 1]
)

@constraint(model, c9[i = 1:length(V), k = 1:length(K)],
    e[i] <= B[i]
)

@constraint(model, c14[i = 1:length(V), k = 1:length(K)],
    B[i] <= l[i]
)

@constraint(model, c15[i = 1:length(P), k = 1:length(K)],
    t[i] <= Lk[i] # HACK n+i
)

@constraint(model, c16[i = 1:length(P), k = 1:length(K)],
    Lk[i] <= l[i]
)

@constraint(model, c11[i = 1:length(V), j = 1:length(V), k = 1:length(K)],
    Qijk[j] >= (Qijk[i] + q[j]) * x[i, j]
)

@constraint(model, c12[i = 1:length(V), k = 1:length(K)],
    max(0, q[i]) <= Qijk[i]
)

@constraint(model, c20[i = 1:length(V), k = 1:length(K)],
    Qijk[i] <= min(Qk, Qk + q[i])
)

@constraint(model, c13[i = 1:length(V), j = 1:length(V), k = 1:length(K)],
    x[i, j] >= 0
)

@constraint(model, c21[i = 1:length(V), j = 1:length(V), k = 1:length(K)],
    x[i, j] <= 1
)

println("The problem has been defined.")
println("Beginning the optimization process...")

optimize!(model)

println("Optimization finished.")

println("Termination status: ", termination_status(model))
println("Primal status: ", primal_status(model))
println("Dual status: ", dual_status(model))

println("Objective value:")
println(objective_value(model))

println("Decision variable value:")
display(value.(x))