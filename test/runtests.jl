include("../src/HydroModelCore.jl")

outputs = [1,2,3]
outputs[[1,2,2], ntuple(_ -> Colon(), 0)...]