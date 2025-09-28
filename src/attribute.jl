"""
    $(TYPEDEF)

Stores attribute information for a hydrological component.

# Fields
$(FIELDS)
"""
struct HydroInfos{IS,OS,SS,PS,NS} <: AbstractInfos
    "Input variable names."
    inputs::IS
    "Output variable names."
    outputs::OS
    "State variable names."
    states::SS
    "Parameter names."
    params::PS
    "Neural network names."
    nns::NS

    function HydroInfos(;
        inputs::AbstractVector{Symbol}=Symbol[],
        outputs::AbstractVector{Symbol}=Symbol[],
        states::AbstractVector{Symbol}=Symbol[],
        params::AbstractVector{Symbol}=Symbol[],
        nns::AbstractVector{Symbol}=Symbol[]
    )
        new{typeof(inputs),typeof(outputs),typeof(states),typeof(params),typeof(nns)}(
            inputs, outputs, states, params, nns
        )
    end
end

"""
    get_input_names(x)

Get input names from `HydroInfos` or `AbstractComponent`.
"""
@inline get_input_names(infos::HydroInfos) = infos.inputs
@inline get_input_names(cpt::AbstractComponent) = get_input_names(cpt.infos)

"""
    get_output_names(x)

Get output names from `HydroInfos` or `AbstractComponent`.
"""
@inline get_output_names(infos::HydroInfos) = infos.outputs
@inline get_output_names(cpt::AbstractComponent) = get_output_names(cpt.infos)

"""
    get_state_names(x)

Get state names from `HydroInfos` or `AbstractComponent`.
"""
@inline get_state_names(infos::HydroInfos) = infos.states
@inline get_state_names(cpt::AbstractComponent) = get_state_names(cpt.infos)

"""
    get_param_names(x)

Get parameter names from `HydroInfos` or `AbstractComponent`.
"""
@inline get_param_names(infos::HydroInfos) = infos.params
@inline get_param_names(cpt::AbstractComponent) = get_param_names(cpt.infos)

"""
    get_nn_names(x)

Get neural network names from `HydroInfos` or `AbstractComponent`.
"""
@inline get_nn_names(infos::HydroInfos) = infos.nns
@inline get_nn_names(cpt::AbstractComponent) = get_nn_names(cpt.infos)

"""
    get_var_names(x)

Get input, output, and state names from `HydroInfos` or `AbstractComponent`.
"""
@inline get_var_names(infos::HydroInfos) = get_input_names(infos), get_output_names(infos), get_state_names(infos)
@inline get_var_names(cpt::AbstractComponent) = get_var_names(cpt.infos)

"""
    get_exprs(cpt)

Get expressions from a flux component.
"""
get_exprs(cpt::AbstractFlux) = cpt.exprs
get_exprs(cpt::AbstractNeuralFlux) = get_outputs(cpt) ~ cpt.chain

"""
    get_var_names(components)

Get unique variable names from a collection of components.
"""
function get_var_names(components::CT) where CT
    inputs, outputs = Vector{Symbol}(), Vector{Symbol}()
    states = reduce(union, get_state_names.(components))
    for comp in components
        tmp_inputs, tmp_outputs = get_input_names(comp), get_output_names(comp)
        tmp_inputs = setdiff(tmp_inputs, outputs)
        union!(inputs, tmp_inputs)
        union!(outputs, tmp_outputs)
    end
    setdiff(inputs, states), outputs, states
end