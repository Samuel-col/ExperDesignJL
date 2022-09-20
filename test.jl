using CSV,DataFrames

cd("/home/samuel/Documentos/ExperDesignJL/")

include("objetos.jl")

# using ExperimentsDesign

df = CSV.File("data/SleepStudyData.csv") |> DataFrame

CSV.read("data/SleepStudyData.csv",DataFrame)

df = dropmissing(df)

mod = Model(df,(true,false,true,true,true),:Tired,:Enough,:Hours,:PhoneReach,:PhoneTime,:Breakfast)
# El primer argumento es el dataframe.
# El segundo argumento es una tupla cuya i-ésima entrada es true si la i-ésima Variable
#  explicativa es un factor y false si no.
# EL tercer argumento es el nombre de la variable dependiente
# Los siguientes argumentos son las variables explicativas. Deben ser tantas como entradas
#  tenga la tupla del segundo argumento.

display(mod.X)

display(mod.names)

θ(mod)

EstimableBase(mod)

YEst(mod)

residuals(mod)


anova1(mod)

anova2(mod)

summary(mod)

summary(mod).σ


# Gráficos

using StatsPlots

gr(minorgrid = true)

begin
    scatter(YEst(mod),residuals(mod),legend = false,
        color = :cyan,shape = :star)
    xlabel!("Y estimado")
    ylabel!("Residuales")
    title!("Supuesto de No Autocorrelación")
end

begin
    boxplot(residuals(mod),legend = false,
        color = :blueviolet,notch = true)
    ylabel!("Residuales")
end




plotResiduals(mod)

plotResiduals(mod;factor = :Tired)

qqplotResiduals(mod)