using JuMP
using CPLEX

println("Packages imported.")

model = Model(CPLEX.Optimizer)

V = [] # conjunto de nós (vértices) do grafo direcionado completo G(V, A) que define oproblema
C = [] # conjunto de nós de coleta
E = [] # conjunto de nós de entrega
P = [] # conjunto de pedidos, cada qual relacionado a um nó de coleta e um de entrega
K = [] # conjunto de veículos disponíveis para roteamento

# Parâmetros
INF = 99999999999999

n = 2 # numero de pedidos a serem servidos
c = [] # custo de transporte do veículo k ao passar pelo arco (i, j)
s = [] # tempo de serviço (embarque ou desembarque) no nó vi ∈ C ∪ E
t = [] # tempo de viagem de transpor o arco (i, j)
q = [] # carregamento do nó vi ∈ C ∪E. É > 0 se vi ∈ C e negativo se vi ∈ C. Além disso, qi = −qn+i
Qk = [] # capacidade do veículo k ∈ K
Tk = [] # tempo max total de rota para cada k ∈ K
e = [] # limite inferior da janela de tempo do nó vi ∈ C ∪ E
l = [] # limite superior da janela de tempo do nó vi ∈ C ∪ E
L = [] # tempo de viagem máxima aceito pelo usuário do pedido i

# Variáveis
@variable(model, B[1:n] >= 0, Int) # instante em que o veículo k inicia o serviço no nó i.
@variable(model, Lk[1:n] >= 0, Int) # tempo de viagem do usuário i quando no veículo k
@variable(model, Qijk[1:n] >= 0, Int) # ocupação do veículo k quando no nó i

# variável de decisão do problema, 1 quando o arco (i, j) é percorrido pelo veículo k e 0 caso contrário.
@variable(model, x[1:10, 1:10] >= 0, Int) # conferir

@objective(model, Min, 
    sum(    
        sum(
            sum(
                c[i, j] * x[i, j] for j in 1:V
            ) for i in 1:V
        ) for k in 1:K
    )
)

print("Objective")

@constraint(model, c1[i = 1:C],
    sum(
        sum(
            x[i, j] for j in 1:V
        ) for k in K
    ) == 1
)

@constraint(model, c2[i = 1:C, k = 1:K],
    sum(x[i, j] for j in 1:V) - sum(x[n + i, j] for j in 1:V) == 0
)

@constraint(model, c3[k = 1:K],
    sum(
        x[0, j] for j in 1:V
    ) == 1
)

@constraint(model, c4[i = 1:union(C, E), k = 1:K],
    sum(x[j, i] for j in 1:V) - sum(x[i, j] for j in 1:V) == 0
)

@constraint(model, c5[k = 1:K],
    sum(
        x[i, 2*n + 1] for i in 1:V
    ) == 1
)

@constraint(model, c6[i = 1:V, j = 1:V, k = 1:K]
    B[j] >= (B[i] + s[i] + t[i, j]) * x[i, j]
)

@constraint(model, c7[i = 1:P, k = 1:K]
    Lk[i] == B[n+i] - (B[i] + s[i])
)

@constraint(model, c8[k = 1:K]
    B[2*n + 1] - B[0] <= Tk
)

@constraint(model, c9[i = 1:V, k = 1:K]
    e[i] <= B[i] && B[i] <= l[i]
)

@constraint(model, c10[i = 1:P, k = 1:K]
    t[i, n+i] <= Lk[i] && Lk[i] <= L[i]
)

@constraint(model, c11[i = 1:V, j = 1:V, k = 1:K]
    Qijk[j] >= (Qijk[i] + q[j]) * x[i, j]
)

@constraint(model, c12[i = 1:V, k = 1:K]
    Max(0, q[i]) <= Qijk[i] && Qijk[i] <= Min(Qk, Qk + q[i])
)

@constraint(model, c13[i = 1:V, j = 1:V, k = 1:K]
    x[i, j] >= 0 && x[i, j] <= 1 
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