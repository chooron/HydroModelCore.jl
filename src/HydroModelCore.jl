module HydroModelCore

using ModelingToolkit: Num
using Symbolics: tosymbol, unwrap

abstract type AbstractComponent end

abstract type AbstractFlux <: AbstractComponent end
abstract type AbstractHydroFlux <: AbstractFlux end
abstract type AbstractNeuralFlux <: AbstractHydroFlux end
abstract type AbstractStateFlux <: AbstractFlux end

abstract type AbstractElement <: AbstractComponent end
abstract type AbstractBucket <: AbstractElement end
abstract type AbstractHydrograph <: AbstractElement end
abstract type AbstractRoute <: AbstractElement end
abstract type AbstractHydroRoute <: AbstractRoute end
abstract type AbstractModel <: AbstractComponent end


include("attribute.jl")
include("display.jl")

export AbstractComponent, AbstractHydroFlux, AbstractNeuralFlux, AbstractStateFlux,
    AbstractElement, AbstractBucket, AbstractHydrograph,
    AbstractRoute, AbstractHydroRoute, AbstractModel

export get_name, get_input_names, get_output_names, get_param_names, get_state_names, get_nn_names
export get_exprs, get_inputs, get_outputs, get_params, get_nns, get_vars

end # module HydroModelCore
