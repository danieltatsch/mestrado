using JuMP
using CPLEX

INF = 99999999999999

println("Packages imported.")

C = [[132, INF, 97, 103] [85, 91, INF, INF] [106, 89, 100, 98] [106, 89, 100, 98]]

N = 4
println("Instance created successfully.")

model = Model(CPLEX.Optimizer)

# criar variáveis
@variable(model, x[1:N, 1:N] >= 0, Int)

@objective(model, Max, 
  sum(
    sum(
      C[i, j] * x[i, j] for j in 1:N
    ) for i in 1:N
  )
)

@constraint(model, c1[j = 1:N],
  sum(
    x[i, j] for i in 1:N
  ) == 1
)

@constraint(model, c2[i = 1:N],
  sum(
    x[i, j] for j in 1:N
  ) <= 1
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