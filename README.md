# HydroModelCore.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Julia Version](https://img.shields.io/badge/julia-v1.0%2B-blue)](https://julialang.org/)
[![version](https://docs.juliahub.com/HydroModelCore/version.svg)](https://juliahub.com/ui/Packages/General/HydroModelCore)

> The foundational framework for building modular hydrological models in Julia

## 📖 Overview

**HydroModelCore.jl** provides the core abstractions, type system, and runtime code generation infrastructure for the [HydroModels.jl](https://github.com/chooron/HydroModels.jl) ecosystem. It enables researchers and developers to build flexible, composable, and high-performance hydrological models through a unified component-based architecture.

## ✨ Key Features

- **🧩 Modular Type System**: Rich abstract type hierarchy for hydrological components (Flux, Bucket, Route, Model)
- **⚡ Runtime Code Generation**: High-performance function building with automatic differentiation support (Zygote-compatible)
- **🔍 Attribute Management**: Comprehensive tools for querying and managing component inputs, outputs, states, and parameters
- **✅ Validation System**: Robust validation with detailed error messages for inputs, parameters, states, and component chains
- **🔬 Symbolic Variables**: Enhanced symbolic system with metadata (descriptions, bounds, units) inspired by [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)
- **📊 Display Utilities**: Professional component inspection and comparison tools

## 🚀 Installation

```julia
using Pkg
Pkg.add("HydroModelCore")
```

Or install from GitHub for the latest development version:

```julia
Pkg.add(url="https://github.com/chooron/HydroModelCore.jl")
```

## 📖 Documentation

- **[API Reference](docs/EXPORTED_FUNCTIONS.md)**: Complete function documentation with examples
- **[Changelog](CHANGELOG.md)**: Version history and release notes
- **[Examples](examples/)**: Working code examples

### Module Overview

| Module | Functions | Description |
|--------|-----------|-------------|
| Attribute System | 15 functions | Access and query component attributes |
| Validation System | 13 functions | Validate inputs, parameters, and component chains |
| Display Utilities | 2 functions | Pretty-print component information |
| Build System | 8+ functions | Generate optimized runtime functions |
| Variable System | 7 functions | Manage symbolic variables with metadata |

## 🔗 Related Projects

- **[HydroModels.jl](https://github.com/chooron/HydroModels.jl)**: Complete hydrological modeling framework
- **[Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl)**: Symbolic computation system
- **[ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl)**: Named component arrays

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
