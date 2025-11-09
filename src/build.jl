"""
Abstract dimension configuration type for compile-time dispatch
"""
abstract type AbstractDimConfig end

"""0D configuration: scalar computation"""
struct Dim0 <: AbstractDimConfig end

"""1D configuration: vector computation"""
struct Dim1 <: AbstractDimConfig end

"""2D configuration: matrix computation"""
struct Dim2 <: AbstractDimConfig end

@inline to_dim_type(::Val{0}) = Dim0()
@inline to_dim_type(::Val{1}) = Dim1()
@inline to_dim_type(::Val{2}) = Dim2()
@inline to_dim_val(::Type{Dim0}) = Val(0)
@inline to_dim_val(::Type{Dim1}) = Val(1)
@inline to_dim_val(::Type{Dim2}) = Val(2)

"""
$(SIGNATURES)

Simplify a Julia expression. For example, convert `:((+)(x, (tanh)(y)))` to `:(x + tanh(y))`.
"""
@inline simplify_expr(expr) = Meta.parse(string(expr))

"""
Generate variable indexing expression based on dimension configuration
"""
@inline index_expr(idx::Int, ::Dim0) = :($idx)
@inline index_expr(idx::Int, ::Dim1) = :($idx, :)
@inline index_expr(idx::Int, ::Dim2) = :($idx, :, :)

"""
$(SIGNATURES)

Generate variable assignment expressions using type dispatch instead of Val parameter.

# Example
```julia
generate_var_assignments([:temp, :prcp], :inputs, Dim1())
# Generates: [:(temp = inputs[1, :]), :(prcp = inputs[2, :])]
```
"""
@inline function generate_var_assignments(
    vars::AbstractVector{Symbol},
    target::Symbol,
    dim_config::AbstractDimConfig;
    prefix::String=""
)
    return [
        :($(Symbol(prefix, var)) = $(target)[$(index_expr(idx, dim_config))])
        for (idx, var) in enumerate(vars)
    ]
end

@inline generate_var_assignments(; vars, target, dims=0, prefix="") = 
    generate_var_assignments(vars, target, to_dim_type(Val(dims)); prefix=prefix)

"""
$(SIGNATURES)

Generate parameter assignment expressions.
"""
@inline function generate_param_assignments(params::AbstractVector{Symbol}; target::Symbol=:pas)
    return [:($p = $(target).params.$p) for p in params]
end

@inline generate_param_assignments(; params, target=:pas) = generate_param_assignments(params; target=target)

"""
$(SIGNATURES)

Generate neural network assignment expressions.
"""
@inline function generate_nn_assignments(nn_fluxes::AbstractVector; target::Symbol=:pas)
    nn_names = [get_nn_names(f)[1] for f in nn_fluxes]
    return [:($nn = $(target).nns.$nn) for nn in nn_names]
end

@inline generate_nn_assignments(; nnfluxes, target=:pas) = generate_nn_assignments(nnfluxes; target=target)

"""
$(SIGNATURES)

Generate all assignment expressions (inputs, states, parameters, neural networks) uniformly.
This is a high-level interface that combines all assignment types together.
"""
@inline function generate_all_assignments(
    infos::HydroModelCore.HydroInfos,
    nn_fluxes::AbstractVector,
    dim_config::AbstractDimConfig
)
    return [
        generate_var_assignments(infos.inputs, :inputs, dim_config)...,
        generate_var_assignments(infos.states, :states, dim_config)...,
        generate_param_assignments(infos.params)...,
        generate_nn_assignments(nn_fluxes)...
    ]
end

@inline generate_all_assignments(infos::HydroModelCore.HydroInfos, nn_fluxes::Vector, dims::Int) =
    generate_all_assignments(infos, nn_fluxes, to_dim_type(Val(dims)))

"""
Generate computation expressions for HydroFlux
"""
@inline function generate_flux_expression(flux::AbstractHydroFlux, ::Dim0)
    return [:($nm = $(toexpr(expr))) for (nm, expr) in zip(get_output_names(flux), flux.exprs)]
end

@inline function generate_flux_expression(flux::AbstractHydroFlux, dim_config::Union{Dim1,Dim2})
    return [:($nm = @. $(simplify_expr(toexpr(expr)))) for (nm, expr) in zip(get_output_names(flux), flux.exprs)]
end

"""
Generate computation expressions for NeuralFlux using unified helper function
"""
@inline function _generate_neural_flux_expr(flux::AbstractNeuralFlux, input_expr::Expr, dim_config::AbstractDimConfig)
    nn_names = get_nn_names(flux)[1]
    nn_inputs = Symbol(nn_names, "_input")
    nn_outputs = Symbol(nn_names, "_output")
    
    return [
        :($(nn_inputs) = $(input_expr) |> $(flux.norm_func)),
        :($(nn_outputs) = $(flux.chain_func)($(nn_inputs), $(nn_names))),
        [:($(nm) = $(nn_outputs)[$(index_expr(i, dim_config))]) for (i, nm) in enumerate(get_output_names(flux))]...
    ]
end

@inline function generate_flux_expression(flux::AbstractNeuralFlux, dim_config::Dim0)
    return _generate_neural_flux_expr(flux, :([$(flux.infos.inputs...)]), dim_config)
end

@inline function generate_flux_expression(flux::AbstractNeuralFlux, dim_config::Dim1)
    return _generate_neural_flux_expr(flux, :(stack([$(flux.infos.inputs...)], dims=1)), dim_config)
end

@inline function generate_flux_expression(flux::AbstractNeuralFlux, dim_config::Dim2)
    return _generate_neural_flux_expr(flux, :(stack([$(flux.infos.inputs...)], dims=1)), dim_config)
end

@inline generate_flux_expression(flux::AbstractFlux, ::Val{D}) where D = 
    generate_flux_expression(flux, to_dim_type(Val(D)))

"""
$(SIGNATURES)

Generate computation call expressions for all fluxes.
"""
@inline function generate_compute_calls(fluxes::AbstractVector, dim_config::AbstractDimConfig)
    return vcat(map(f -> generate_flux_expression(f, dim_config), fluxes)...)
end

@inline generate_compute_calls(; fluxes, dims=0) = 
    generate_compute_calls(fluxes, to_dim_type(Val(dims)))

"""
$(SIGNATURES)

Generate return expression for state differentials.

# Arguments
- `dfluxes`: Vector of state fluxes
- `broadcast`: Whether to use broadcasting (for multiple grids)
"""
@inline function generate_states_return_expr(dfluxes::AbstractVector, broadcast::Bool)
    all_exprs = reduce(vcat, get_exprs.(dfluxes))
    
    if broadcast
        broadcast_exprs = [:(@. $(simplify_expr(toexpr(expr)))) for expr in all_exprs]
        return :(return vcat($(broadcast_exprs...)))
    else
        plain_exprs = [toexpr(expr) for expr in all_exprs]
        return :(return [$(plain_exprs...)])
    end
end

@inline generate_states_expression(; dfluxes, broadcast=false) = 
    generate_states_return_expr(dfluxes, broadcast)

"""
Configuration for building runtime functions
"""
struct FunctionBuildConfig
    signature::Expr
    assignments::Vector{Any}
    computations::Vector{Any}
    return_expr::Expr
end

"""
$(SIGNATURES)

Build RuntimeGeneratedFunction from configuration.
This is a generic function builder that converts configuration into actual function expression.
"""
function build_runtime_function(config::FunctionBuildConfig)
    func_body = Expr(:block,
        :(Base.@_inline_meta),
        config.assignments...,
        config.computations...,
        config.return_expr
    )
    
    func_expr = Expr(:function, config.signature, func_body)
    
    return @RuntimeGeneratedFunction(func_expr)
end

"""
$(SIGNATURES)

Build a pair of functions (flux and diff) with unified logic.
- If has states: returns both functions
- Otherwise: returns flux function and dummy diff function
"""
function build_function_pair(
    flux_config::FunctionBuildConfig,
    diff_config::Union{FunctionBuildConfig, Nothing},
    has_states::Bool
)
    flux_func = build_runtime_function(flux_config)
    
    if has_states && !isnothing(diff_config)
        diff_func = build_runtime_function(diff_config)
        return flux_func, diff_func
    else
        return flux_func, (_) -> nothing
    end
end

"""
$(TYPEDSIGNATURES)

Build simple flux function from symbolic expressions.
"""
function build_flux_func(exprs::Vector{Num}, infos::HydroModelCore.HydroInfos)
    flux_exprs = [:(@. $(simplify_expr(toexpr(expr)))) for expr in exprs]
    
    config = FunctionBuildConfig(
        :((inputs, pas)),
        [
            generate_var_assignments(infos.inputs, :inputs, Dim0())...,
            generate_param_assignments(infos.params)...
        ],
        [:($o = $expr) for (o, expr) in zip(infos.outputs, flux_exprs)],
        :(return [$((infos.outputs)...)])
    )
    
    return build_runtime_function(config)
end

"""
$(TYPEDSIGNATURES)

Build bucket function with unified interface using type system and generic framework.
"""
function build_bucket_func(
    fluxes::Vector{<:AbstractFlux},
    dfluxes::Vector{<:AbstractStateFlux},
    infos::HydroModelCore.HydroInfos,
    multiply::Bool
)
    nn_fluxes = filter(f -> f isa AbstractNeuralFlux, fluxes)
    has_states = !isempty(infos.states)
    
    # Select dimension configuration based on multiply mode
    if multiply
        flux_dim = Dim2()  # Multiple grids: flux uses 2D
        diff_dim = Dim1()  # diff uses 1D + broadcast
    else
        flux_dim = Dim1()  # Single grid: flux uses 1D
        diff_dim = Dim0()  # diff uses 0D
    end
    
    # Build flux function configuration
    flux_config = FunctionBuildConfig(
        :((inputs, states, pas)),
        generate_all_assignments(infos, nn_fluxes, flux_dim),
        generate_compute_calls(fluxes, flux_dim),
        :(return [$((infos.outputs)...)])
    )
    
    # Build diff function configuration (if needed)
    # Note: diff_dim != Dim0() implies broadcast mode
    diff_config = if has_states
        FunctionBuildConfig(
            :((inputs, states, pas)),
            generate_all_assignments(infos, nn_fluxes, diff_dim),
            generate_compute_calls(fluxes, diff_dim),
            generate_states_return_expr(dfluxes, diff_dim != Dim0())
        )
    else
        nothing
    end
    
    return build_function_pair(flux_config, diff_config, has_states)
end

"""
$(TYPEDSIGNATURES)

Build routing function for river network calculations.
- flux function: 2D assignment + 1D computation
- diff function: 1D assignment + 1D computation, with special return format
"""
function build_route_func(
    fluxes::Vector{<:AbstractHydroFlux},
    dfluxes::Vector{<:AbstractStateFlux},
    infos::HydroModelCore.HydroInfos
)
    nn_fluxes = filter(f -> f isa AbstractNeuralFlux, fluxes)
    has_states = !isempty(infos.states)
    
    flux_config = FunctionBuildConfig(
        :((inputs, states, pas)),
        generate_all_assignments(infos, nn_fluxes, Dim2()),
        generate_compute_calls(fluxes, Dim1()),
        :(return [$(infos.outputs...)])
    )
    
    diff_config = if has_states
        # Route diff has special return format: [outputs..., state_diffs...]
        state_diff_exprs = [:(@. $(simplify_expr(toexpr(expr)))) 
                           for expr in reduce(vcat, get_exprs.(dfluxes))]
        
        FunctionBuildConfig(
            :((inputs, states, pas)),
            generate_all_assignments(infos, nn_fluxes, Dim1()),
            generate_compute_calls(fluxes, Dim1()),
            :([$(infos.outputs...), vcat($(state_diff_exprs...))])
        )
    else
        nothing
    end
    
    return build_function_pair(flux_config, diff_config, has_states)
end

"""
$(TYPEDSIGNATURES)

Build Unit Hydrograph (UH) function.

Generates two functions:
1. uh_func(t, pas): compute weight based on time t
2. max_lag_func(pas): compute maximum lag time
"""
function build_uh_func(
    uh_conds::AbstractVector{<:Pair},
    params::AbstractVector{Symbol},
    max_lag::Number
)
    conditions_rev = vcat([0], reverse(first.(uh_conds)))
    values_rev = reverse(last.(uh_conds))
    param_assigns = generate_param_assignments(params)
    
    condition_checks = map(eachindex(values_rev)) do i
        :(
            if $(toexpr(conditions_rev[i])) ≤ t ≤ $(toexpr(conditions_rev[i+1]))
                return $(toexpr(values_rev[i]))
            end
        )
    end
    
    uh_func_expr = :(function (t, pas)
        $(param_assigns...)
        $(condition_checks...)
        return 1.0
    end)
    
    max_lag_expr = :(function (pas)
        $(param_assigns...)
        return ceil($(toexpr(max_lag)))
    end)
    
    return (
        @RuntimeGeneratedFunction(uh_func_expr),
        @RuntimeGeneratedFunction(max_lag_expr)
    )
end

"""
$(SIGNATURES)

Preview generated function expression for debugging.
"""
function preview_function_expr(config::FunctionBuildConfig)
    func_body = Expr(:block,
        :(Base.@_inline_meta),
        config.assignments...,
        config.computations...,
        config.return_expr
    )
    
    func_expr = Expr(:function, config.signature, func_body)
    
    println("Generated function expression:")
    println("=" ^ 80)
    println(func_expr)
    println("=" ^ 80)
end