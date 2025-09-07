module HydroModelCore

using DocStringExtensions
using Symbolics
using Symbolics: tosymbol, unwrap, wrap, Num, Symbolic, @variables, get_variables

abstract type AbstractComponent end
abstract type AbstractNetwork end

abstract type AbstractFlux <: AbstractComponent end
abstract type AbstractHydroFlux <: AbstractFlux end
abstract type AbstractNeuralFlux <: AbstractHydroFlux end
abstract type AbstractStateFlux <: AbstractFlux end

abstract type AbstractElement <: AbstractComponent end
abstract type AbstractBucket <: AbstractElement end
abstract type AbstractHydroBucket <: AbstractBucket end
abstract type AbstractNeuralBucket <: AbstractBucket end
abstract type AbstractHydrograph <: AbstractElement end
abstract type AbstractRoute <: AbstractElement end
abstract type AbstractHydroRoute <: AbstractRoute end
abstract type AbstractModel <: AbstractComponent end


export AbstractComponent, # base type
    AbstractHydroFlux, AbstractNeuralFlux, AbstractStateFlux, # flux types
    AbstractElement, # element types
    AbstractHydroBucket, AbstractNeuralBucket, # bucket types
    AbstractHydrograph, # hydrograph types
    AbstractRoute, AbstractHydroRoute, # route types
    AbstractModel, # model types
    AbstractNetwork # network types

include("attribute.jl")
include("check.jl")
include("display.jl")
include("parameters.jl")
include("variables.jl")

export get_input_names, get_output_names, get_param_names, get_state_names, get_nn_names, get_exprs
export @variables, @parameters
export getdescription, getbounds, getunit, getguess

end # module HydroModelCore
