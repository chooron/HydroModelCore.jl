# HydroModelCore.jl: Core Abstractions for Hydrological Modeling

A Julia package that provides core abstractions and functionality for hydrological modeling. This package is a component of [HydroModels.jl](https://github.com/chooron/HydroModels.jl).

## Overview

HydroModelCore.jl defines the fundamental abstract types and their attribute access methods used in hydrological modeling. It provides a flexible and extensible framework for building hydrological models.

## Core Abstractions

```
AbstractComponent
├── AbstractModel
│   └── AbstractHydroModel
├── AbstractElement
│   ├── AbstractBucket
│   ├── AbstractHydrograph
│   └── AbstractRoute
│       └── AbstractHydroRoute
└── AbstractFlux
    ├── AbstractHydroFlux
    │   └── AbstractNeuralFlux
    └── AbstractStateFlux
```

## Features

- **Component-based Architecture**: Modular design allowing flexible composition of hydrological models
- **Rich Attribute System**: Access to various component attributes:
  - Name (`get_name` for get component name)
  - Inputs (`get_input_names` for Symbol type and `get_inputs` for Num type)
  - Outputs (`get_output_names` for Symbol type and `get_outputs` for Num type)
  - Parameters (`get_param_names` for Symbol type and `get_params` for Num type)
  - States (`get_state_names` for Symbol type and `get_states` for Num type)
  - Neural Networks (`get_nn_names` and `get_nns` for Num type)
  - Expressions (`get_exprs`)
- **Pretty Printing**: Customized display formats for all component types
- **Type Safety**: Strong type system ensuring model consistency

## Installation

```julia
using Pkg
Pkg.add("HydroModelCore")
```
