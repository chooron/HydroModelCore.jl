module HydroModelCore

using DocStringExtensions
using ComponentArrays
using Symbolics
using Symbolics: tosymbol, unwrap, wrap, Num, Symbolic, @variables, get_variables

abstract type AbstractComponent end
abstract type AbstractNetwork end

abstract type AbstractFlux <: AbstractComponent end
abstract type AbstractHydroFlux <: AbstractFlux end
abstract type AbstractNeuralFlux <: AbstractFlux end
abstract type AbstractStateFlux <: AbstractFlux end

abstract type AbstractElement <: AbstractComponent end
abstract type AbstractBucket <: AbstractElement end
abstract type AbstractHydroBucket <: AbstractBucket end
abstract type AbstractNeuralBucket <: AbstractBucket end
abstract type AbstractRoute <: AbstractElement end
abstract type AbstractHydroRoute <: AbstractRoute end
abstract type AbstractHydrograph <: AbstractRoute end
abstract type AbstractModel <: AbstractComponent end


export AbstractComponent, # base type
    AbstractFlux, AbstractHydroFlux, AbstractNeuralFlux, AbstractStateFlux, # flux types
    AbstractElement, # element types
    AbstractBucket, AbstractHydroBucket, AbstractNeuralBucket, # bucket types
    AbstractRoute, AbstractHydroRoute, AbstractHydrograph, # route types
    AbstractModel, # model types
    AbstractNetwork # network types

include("attribute.jl")
include("check.jl")
include("display.jl")
include("parameters.jl")
include("variables.jl")

export HydroInfos
export get_input_names, get_output_names, get_param_names, get_state_names, get_nn_names, get_exprs, get_var_names
export @variables, @parameters, isparameter
export getdescription, getbounds, getunit, getguess

end # module HydroModelCore
