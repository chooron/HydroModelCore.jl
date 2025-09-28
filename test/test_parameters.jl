include("../src/HydroModelCore.jl")

HydroModelCore.@parameters p [bounds=(0, 1), description="test", guess=0.5,unit="m"]

@info HydroModelCore.getbounds(p)
@info HydroModelCore.getdescription(p)
@info HydroModelCore.getguess(p)
@info HydroModelCore.getunit(p)