module PyUC

using UnitCommitment
using JuMP
using Cbc
using JSON
using Gurobi
using MathOptInterface
const MOI = MathOptInterface

const INSTANCE = Ref{Any}(nothing)
const GEN_NAMES = Ref{Vector{String}}(String[])
const T_HORIZON = Ref{Int}(0)
const MODEL = Ref{Union{JuMP.Model, Nothing}}(nothing) # Global model that we will modify with fixes
const LAST_FIXES = Ref{Vector{Tuple{String,Int}}}([])
const PMAX = Ref{Dict{String, Float64}}(Dict())
const DEMAND_T = Ref{Vector{Float64}}([])

"Load a benchmark instance (e.g., matpower/case14/2017-01-01) and cache metadata."
function load_instance(name::AbstractString)

    instance = UnitCommitment.read_benchmark(name)
    gens = instance.scenarios[1].thermal_units

    PMAX[] = Dict(
        u.name => (isa(u.max_power, AbstractVector) ? maximum(u.max_power) : u.max_power)
        for u in gens
    )

    # Demand per time step
    scenario = instance.scenarios[1]
    buses = scenario.buses   # ← capture locally

    DEMAND_T[] = [
        sum(bus.load[t] for bus in buses)
        for t in 1:instance.time
    ]

    INSTANCE[]  = instance
    GEN_NAMES[] = [u.name for u in gens]
    T_HORIZON[] = instance.time

    println("Building UC model (one-time cost)...")

    model = UnitCommitment.build_model(
        instance = instance,
        optimizer = optimizer_with_attributes(
            Gurobi.Optimizer,
            "OutputFlag" => 0,
            "Threads" => 8,
            "Presolve" => 2,
            "MIPGap" => 0.01,     # relax for RL speed
            "MIPFocus" => 1,
            "NumericFocus" => 2
        )
    )

    MODEL[] = model

    return (GEN_NAMES[], T_HORIZON[])
end

"Internal: build a JuMP model and fix 'is_on[g,t]' for provided tuples."
function _build_model_with_fixes(fixes)::JuMP.Model
    instance = INSTANCE[]

    model = UnitCommitment.build_model(
    instance = instance,
    optimizer = optimizer_with_attributes(
        Gurobi.Optimizer,
        "Presolve" => 0,
        "DualReductions" => 0,
        "NumericFocus" => 3,
        "MIPFocus" => 1,
        "Threads" => 0
        )
    )
          
    for f in fixes
        g = String(f[1]); t = Int(f[2]); on = Bool(f[3] != 0)
        JuMP.fix(model[:is_on][g, t], on ? 1.0 : 0.0, force=true)
    end
    return model
end

# Internal: clear previous fixes
function _clear_fixes!()
    """Unfix all previously fixed variables based on the LAST_FIXES list. This modifies the global MODEL in-place.
    """

    model = MODEL[]

    for (g,t) in LAST_FIXES[]
        JuMP.unfix(model[:is_on][g,t])
    end

    empty!(LAST_FIXES[])
end

# Internal: apply new fixes
function _apply_fixes!(fixes)
    """"Apply fixes to the model based on the provided list of (generator, time, on/off) tuples. This modifies the global MODEL in-place.
    """

    model = MODEL[]

    max_fixes = min(length(fixes), 20)   # ← tune this (10–50)

    for i in 1:max_fixes
        f = fixes[i]

        g = String(f[1])
        t = Int(f[2])
        on = Bool(f[3] != 0)

        JuMP.fix(model[:is_on][g,t], on ? 1.0 : 0.0; force=true)
        push!(LAST_FIXES[], (g,t))
    end
end

# Warm start (critical for speed)
function _warm_start!()
    """"Set start values for all variables based on the last solution. This is crucial for solver speed when applying fixes iteratively.
    """
    model = MODEL[]

    if JuMP.primal_status(model) in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
        for v in all_variables(model)
            JuMP.set_start_value(v, JuMP.value(v))
        end
    end
end

function _feasibility_filter(fixes)

    gen_available = Dict{Int, Float64}()

    for (g, t, val) in fixes
        if val == 1
            gen_available[t] = get(gen_available, t, 0.0) + PMAX[][g]
        end
    end

    # Check ALL time periods
    for t in 1:T_HORIZON[]
        available = get(gen_available, t, 0.0)

        if available < 0.85 * DEMAND_T[][t]
            return false
        end
    end

    return true
end

# Main solver call
function solve_with_fixes(fixes)

    model = MODEL[]
    if model === nothing
        error("Model not initialized. Call load_instance() first.")
    end

    _clear_fixes!()
    _apply_fixes!(fixes)
    _warm_start!()

    if !_feasibility_filter(fixes)
        penalty = 0.0

        for t in 1:T_HORIZON[]
            available = 0.0

            for (g, tt, val) in fixes
                if tt == t && val == 1
                    available += PMAX[][g]
                end
            end

            deficit = max(0.0, DEMAND_T[][t] - available)
            penalty += deficit
        end

        return 1e6 + 1000 * penalty, 0.0
    end

    try
        UnitCommitment.optimize!(model)
    catch e
        @warn "Solver exception" exception=(e, catch_backtrace())
        penalty = 0.0

        for t in 1:T_HORIZON[]
            available = 0.0

            for (g, tt, val) in fixes
                if tt == t && val == 1
                    available += PMAX[][g]
                end
            end

            deficit = max(0.0, DEMAND_T[][t] - available)
            penalty += deficit
        end

        return 1e6 + 1000 * penalty, 0.0
    end

    term_status = JuMP.termination_status(model)
    primal_status = JuMP.primal_status(model)
    result_count = MOI.get(model, MOI.ResultCount())

    has_solution =
        (term_status in (MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.FEASIBLE_POINT)) &&
        (primal_status in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)) &&
        (result_count > 0)

    if !has_solution
        @warn "No feasible solution" term_status primal_status result_count
        return 1e6, 0.0   # ← consistent penalty
    end

    obj = JuMP.objective_value(model)

    # NOTE: SolveTimeSec is safe even if no solution, but we already checked
    solve_time = try
        MOI.get(model, MOI.SolveTimeSec())
    catch
        0.0
    end

    return obj, solve_time
end

"Return MILP optimal cost for comparison."
function get_MILP_cost(instance_name::AbstractString)
    instance = UnitCommitment.read_benchmark(instance_name)

    model = UnitCommitment.build_model(
    instance = instance,
    optimizer = optimizer_with_attributes(
        Gurobi.Optimizer,
        "Presolve" => 0,
        "DualReductions" => 0,
        "NumericFocus" => 3,
        "MIPFocus" => 1,
        "Threads" => 0
        )
    )

    UnitCommitment.optimize!(model)
    status = JuMP.termination_status(model)
    if status != MOI.OPTIMAL
        @warn "Unit commitment did not solve to optimality. Status: $status"
        return nothing
    end
    return JuMP.objective_value(model)
end

end # module