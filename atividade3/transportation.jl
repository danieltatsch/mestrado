using JuMP
using CPLEX

INF = 99999999999999

println("Packages imported.")

supplies = [135, 56, 93]
demands = [62, 83, 39, 91]

# costs = [[132, 85, 106] [INF, 91, 89] [97, INF, 100] [103, INF, 98]]

# C = [[132, INF, 97, 103] [85, 91, INF, INF] [106, 89, 100, 98]]
C = [[132, 85, 106] [INF, 91, 89] [97, INF, 100] [103, INF, 98]]

M = length(supplies)
N = length(demands)

println("Instance created successfully.")

model = Model(CPLEX.Optimizer)

# criar variáveis
@variable(model, x[1:M, 1:N] >= 0, Int)

@objective(model, Min, 
  sum(
    sum(
      C[i, j] * x[i, j] for j in 1:N
    ) for i in 1:M
  )
)

@constraint(model, demands_statisfied[j = 1:N],
  sum(
    x[i, j] for i in 1:M
  ) == demands[j]
)

@constraint(model, below_capacity[i = 1:M],
  sum(
    x[i, j] for j in 1:N
  ) <= supplies[i]
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