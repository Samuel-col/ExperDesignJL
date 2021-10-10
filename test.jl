using CSV,DataFrames

cd("/Archivos/A/Julia/DiseñoDeExperimentos/")

include("objetos.jl")

# using ExperimentsDesign

df = CSV.File("data/SleepStudyData.csv") |> DataFrame

df = dropmissing(df)

mod = Model(df,(true,false,true,true,true),:Tired,:Enough,:Hours,:PhoneReach,:PhoneTime,:Breakfast)

display(mod.X)

display(mod.names)

θ(mod)

EstimableBase(mod)

YEst(mod)

residuals(mod)


anova1(mod)
