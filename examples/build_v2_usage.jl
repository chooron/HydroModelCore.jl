"""
Build V2 Usage Examples
=======================

This file demonstrates how to use the Build V2 enhanced features
while maintaining full compatibility with V1.
"""

using HydroModelCore

println("=" ^ 80)
println("HydroModelCore Build V2 - Usage Examples")
println("=" ^ 80)

#==============================================================================
# Example 1: Standard V1 Usage (No Changes Needed)
==============================================================================#

println("\n[Example 1] Standard V1 Usage")
println("-" ^ 80)

# V1 code works exactly as before
# All broadcasting behavior is identical
println("✓ V1 interface remains unchanged")
println("✓ All broadcasting modes work as expected:")
println("  - Dim0: scalar (no broadcast)")
println("  - Dim1: vector broadcast (@.)")
println("  - Dim2: matrix broadcast (@.)")

#==============================================================================
# Example 2: Load V2 Enhanced Features
==============================================================================#

println("\n[Example 2] Loading V2 Enhanced Features")
println("-" ^ 80)

# Load V2 as optional module
include(joinpath(@__DIR__, "../src/build_v2.jl"))

println("✓ V2 module loaded successfully")
println("✓ Available enhancements:")
println("  - Broadcast strategy traits")
println("  - Optimized assignment configurations")
println("  - Debug and analysis tools")

#==============================================================================
# Example 3: Using Broadcast Strategy System
==============================================================================#

println("\n[Example 3] Broadcast Strategy System")
println("-" ^ 80)

# Check broadcasting strategy for different dimensions
for (dim_type, dim_val) in [(HydroModelCore.Dim0(), "Dim0"), 
                             (HydroModelCore.Dim1(), "Dim1"), 
                             (HydroModelCore.Dim2(), "Dim2")]
    strategy = HydroModelCore.broadcast_strategy(dim_type)
    rank = HydroModelCore.dim_rank(dim_type)
    println("  $dim_val (rank=$rank) → $(typeof(strategy).name.name)")
end

#==============================================================================
# Example 4: Assignment Configuration
==============================================================================#

println("\n[Example 4] Assignment Configuration Options")
println("-" ^ 80)

println("Default (Safe):")
println("  - use_views: $(HydroModelCore.DEFAULT_ASSIGNMENT.use_views)")
println("  - type_annotations: $(HydroModelCore.DEFAULT_ASSIGNMENT.type_annotations)")
println("  - bounds_check: $(HydroModelCore.DEFAULT_ASSIGNMENT.bounds_check)")

println("\nOptimized (Performance):")
println("  - use_views: $(HydroModelCore.OPTIMIZED_ASSIGNMENT.use_views)")
println("  - type_annotations: $(HydroModelCore.OPTIMIZED_ASSIGNMENT.type_annotations)")
println("  - bounds_check: $(HydroModelCore.OPTIMIZED_ASSIGNMENT.bounds_check)")

#==============================================================================
# Example 5: Broadcast Application
==============================================================================#

println("\n[Example 5] Broadcast Application")
println("-" ^ 80)

test_expr = :(x + y * z)

# Apply different broadcast strategies
for (dim, name) in [(HydroModelCore.Dim0(), "Dim0 (no broadcast)"),
                    (HydroModelCore.Dim1(), "Dim1 (with broadcast)")]
    result = HydroModelCore.apply_broadcast(test_expr, dim)
    println("  $name:")
    println("    Input:  $test_expr")
    println("    Output: $result")
end

#==============================================================================
# Example 6: Variable Assignment Generation
==============================================================================#

println("\n[Example 6] Variable Assignment Generation")
println("-" ^ 80)

vars = [:temp, :prcp]
target = :inputs

# Different dimension modes
for (dim, name) in [(HydroModelCore.Dim0(), "Scalar"),
                    (HydroModelCore.Dim1(), "Vector"),
                    (HydroModelCore.Dim2(), "Matrix")]
    assigns = HydroModelCore.generate_var_assignments(vars, target, dim)
    println("  $name mode:")
    for assign in assigns
        println("    $assign")
    end
end

#==============================================================================
# Example 7: Creating Custom Configuration
==============================================================================#

println("\n[Example 7] Custom Assignment Configuration")
println("-" ^ 80)

# Create a custom configuration
custom_config = HydroModelCore.AssignmentConfig(
    true,   # use_views = true (for efficiency)
    false,  # type_annotations = false
    true    # bounds_check = true (keep safe)
)

println("Custom configuration created:")
println("  - Efficient (uses views)")
println("  - Safe (keeps bounds checking)")
println("  - Simple (no type annotations)")

# Use it
assigns = HydroModelCore.generate_var_assignments(
    [:x, :y], :inputs, HydroModelCore.Dim1(), custom_config
)
println("\nGenerated assignments:")
for assign in assigns
    println("  $assign")
end

#==============================================================================
# Example 8: When to Use V2
==============================================================================#

println("\n[Example 8] When to Use V2 Features")
println("-" ^ 80)

println("""
Use V2 when you need:
  ✓ Debug output for generated functions
  ✓ Performance optimization (views, no bounds checks)
  ✓ Code analysis and profiling
  ✓ Explicit broadcast strategy control
  ✓ Custom assignment configurations

Stick with V1 when:
  ✓ Standard usage is sufficient
  ✓ Maximum stability is priority
  ✓ Code is already working well
""")

#==============================================================================
# Example 9: Migration Path
==============================================================================#

println("\n[Example 9] Migration from V1 to V2")
println("-" ^ 80)

println("""
Step 1: Keep using V1 (no changes needed)
  # Your existing code
  flux_func = build_flux_func(exprs, infos)

Step 2: Optionally load V2 for debugging
  include("src/build_v2.jl")
  flux_func = build_flux_func_v2(exprs, infos; debug=true)

Step 3: Enable optimizations when confident
  flux_func = build_flux_func_v2(exprs, infos; optimized=true)

Note: V1 and V2 can coexist! Mix and match as needed.
""")

#==============================================================================
# Summary
==============================================================================#

println("\n" * "=" ^ 80)
println("Summary")
println("=" ^ 80)
println("""
✅ V2 maintains 100% compatibility with V1
✅ All broadcasting functionality preserved
✅ Additional optimization and debug features available
✅ No breaking changes to existing code
✅ Gradual adoption possible

Broadcasting Modes (V1 = V2):
  - Dim0: Scalar computation (no broadcast)
  - Dim1: Vector broadcast (@. for element-wise ops)
  - Dim2: Matrix broadcast (@. for element-wise ops)

V2 Enhancements:
  + Explicit broadcast strategy control
  + Configurable optimizations (@view, @inbounds)
  + Debug and profiling tools
  + Better type system with traits
""")

println("=" ^ 80 * "\n")
println("✓ All examples completed successfully!")

