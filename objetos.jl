using LinearAlgebra, Distributions, DataFrames, StatsPlots

import Base: summary
import LinearAlgebra: rank
import Base: display

struct Factor # Representa factores
    val::Array{Int64,1} # Valores del vector
    lev::Array{Any,1} # Los niveles que representan los valores asignados

    # Método Constructivo
    function Factor(x)
        levels = x |> unique |> sort #  Vector de niveles
        
        convert = Dict(zip(levels,1:length(levels))) # Diccionario de conversión

        formated = get.(Ref(convert),x,0) 

        new(formated,levels) # Creación del objeto
    end
end

function display(f::Factor)
    dump(f)
end

struct Model
    y::Array{Real,1} # Variable respuesta
    X::Array{<:Real,2} # Matriz diseño
    Xi::Array{Any,1} # Matrices asociadas a cada factor
    names::Tuple{Vararg{Symbol}} # Tupla de nombres

    # Método Constructivo
    function Model(data::DataFrame,factor::Tuple{Vararg{Bool}},names...)
        n = nrow(data)
        X = ones(n,1) # Intercepto
        Xi = []
        push!(Xi,X)
        for i in 2:length(names)
            if !factor[i-1]
                X = hcat(X,data[:,names[i]])
                push!(Xi,data[:,names[i]])
            else
                fac = Factor(data[:,names[i]])

                tmp = hcat([fac.val .== l for l in 1:length(fac.lev)]...)

                X = hcat(X,tmp)

                push!(Xi,Matrix(tmp))
            end
        end
        y = data[:,names[1]]
        new(y,X,Xi,names)
    end
end

function display(m::Model)
    dump(m)
end

function gInvS(A::Matrix)
    if A'!=A
        error("La matriz no es simétrica.")
    end
    l = size(A)[1]
    cols = (1:l)[abs.(diag(qr(A).R)).>1e-8]
    GInv = zeros(l,l)
    GInv[cols,cols] = inv(A[cols,cols])
    return GInv
end

function gInvS(A::Real)
    return 1/A
end

function proy(X)
    return X*gInvS(X'*X)*X'
end

function EstimableBase(mod::Model)
    return round.(gInvS(mod.X'*mod.X)*mod.X'*mod.X,digits=5)
end

function θ(mod::Model;inversa::Bool=false)
    if inversa
        return gInvS(mod.X'*mod.X)*mod.X'*mod.y, gInvS(mod.X'*mod.X)
    else
        return gInvS(mod.X'*mod.X)*mod.X'*mod.y
    end
end

function YEst(mod::Model)
    return mod.X*θ(mod)
end

function residuals(mod::Model)
    return mod.y - YEst(mod)
end


#--- Anova tipo 1

function SC1(mod::Model)
    X1 = []
    push!(X1,mod.Xi[1])
    for k in 2:length(mod.Xi)
        push!(X1,hcat(X1[end],mod.Xi[k]))
    end
    push!(X1,zeros(length(mod.y)) .+ I(length(mod.y)))

    sc = []
    for k in 2:length(X1)
        push!(sc,mod.y'*(proy(X1[k])-proy(X1[k-1]))mod.y)
    end
    push!(sc,mod.y'*(proy(X1[end])-proy(X1[1]))mod.y)
    return tuple(sc...) # Los primeros k valores son las sumas de cuadrados tipo 1 cuya suma es la suma de cuadrados de regresión, el penúltimo es la suma de cuadrados de los residuales y el último valor es la suma de cuadrados total.
end


function rank(V::Array{<:Real,1})
    return 1
end


function gl1(mod::Model)

    # Grados de libertad de la regresión
    gl = [rank(xi) == 1 ? 1 : rank(xi) - 1 for xi in mod.Xi[2:end]]

    push!(gl,length(mod.y)-rank(mod.X)) # Grados de libertad de los residuales
    push!(gl,length(mod.y)-1) # Grados de libertad totales
    return tuple(gl...)
end

function CM1(mod::Model)
    cm = []
    sc = SC1(mod)
    gl = gl1(mod)
    
    return sc ./ gl
end

function FStat1(mod::Model)
    fStat = []
    cm = CM1(mod)
    
    return cm[1:(end-2)] ./ cm[end - 1]
end

function pValue1(mod::Model)
    gl = gl1(mod)
    fStats = FStat1(mod)

    pv = [ccdf(FDist(gl[i],gl[end-1]),fStats[i]) for i in 1:length(fStats)]

    return tuple(pv...)
end

function regAn(mod::Model)
    sc = SC1(mod)
    cm = CM1(mod)
    SCReg = sum(sc[1:(end-2)])
    gl = rank(mod.X)-1
    CMReg = SCReg/gl
    fStat = CMReg/cm[end-1]
    n = length(mod.y)
    pVal = ccdf(FDist(gl,n-gl+1),fStat)

    return (SCReg,gl,CMReg,fStat,pVal)
end

function anova1(mod::Model)
    ra = regAn(mod)
    sc = SC1(mod)
    cm = CM1(mod)
    gl = gl1(mod)
    fs = FStat1(mod)
    pv = pValue1(mod)
    table = DataFrame(
        VarationCause = ["Model",(string.(mod.names[2:end]))...,"Residuals","Total"],
        df = [ra[2],gl...],
        SS = [ra[1],sc...],
        MS = [ra[3],cm...],
        F  = [ra[4],fs...,NaN,NaN],
        pValue = [ra[5],pv...,NaN,NaN])
    return table
end

#--- Anova tipo II

function SC2(mod::Model)
    X2 = []
    for i = 2:length(mod.Xi)
        tmp = mod.Xi[1]
        for j = 2:length(mod.Xi)
            if i!=j
                tmp = hcat(tmp,mod.Xi[j])
            end
        end
        push!(X2,tmp)
    end
    sc = []
    for Xp in X2
        push!(sc,mod.y'*(proy(mod.X)-proy(Xp))*mod.y)
    end
    return tuple(sc...)
end

function gl2(mod::Model)
    
    gl = [rank(xi) == 1 ? 1 : rank(xi) - 1 for xi in mod.Xi[2:end]]

    push!(gl,length(mod.y)-rank(mod.X)) # Grados de libertad de los residuales
    return tuple(gl...)
end

function CM2(mod::Model)

    sc = SC2(mod)
    gl = gl2(mod)

    return sc ./ gl[1:(end-1)]
end

function FStat2(mod::Model)

    cm = CM2(mod)
    σ = CM1(mod)[end-1]
    
    return cm ./ σ
end

function pValue2(mod::Model)

    gl = gl2(mod)
    fStats = FStat2(mod)
    
    pv = [ccdf(FDist(gl[i],gl[end]),fStats[i]) for i in 1:length(fStats)]
    
    return tuple(pv...)
end

function anova2(mod::Model)
    sc = SC2(mod)
    cm = CM2(mod)
    gl = gl2(mod)
    fs = FStat2(mod)
    pv = pValue2(mod)
    table = DataFrame(
        Variable=[(string.(mod.names[2:end]))...],
        df = [gl[1:(end-1)]...],
        SS = [sc...],
        MS = [cm...],
        F = [fs...],
        pValue = [pv...])
    return table
end


#--- Otras funciones


struct ModelSummary  # Resumen de un modelo
    σ::Float64       # Varianza de los residuales
    R²::Float64      # Coeficiente de Determinación
    R²adj::Float64   # Coeficiente de Determinación ajustado
    Fstat::Float64   # Estadística F
    pValue::Float64  # Valor p
    N_Obs::Int       # Número de observaciones
    N_Var::Int       # Número de variables (explicativas)
    regDF::Int       # Grados de libertad del modelo


    # Método Constructivo
    function ModelSummary(x)
        σ = CM1(mod)[end-1]
        sc = SC1(mod)
        cm = CM1(mod)
        regression = regAn(mod)
        R = regression[1]/sc[end]
        R_adj = 1 - σ/cm[end]
        F = regression[end-1]
        pVal = regression[end]
        n = length(mod.y)
        k = length(mod.names)-1
        gl = rank(mod.X)

        new(σ,R,R_adj,F,pVal,n,k,gl)
    end
end

function display(s::ModelSummary)
    dump(s)
end

function summary(m::Model)
    return ModelSummary(m)
end


#--- Gráficos

function plotResiduals(mod::Model;factor::Union{Symbol,Missing} = missing)
    
    significativeVariables = anova2(mod).pValue .< 0.05
    my_factors = [ typeof(M) <: AbstractVector ? 1 : size(M)[2] for M in mod.Xi] .> 1
    significativeFactors = significativeVariables .* my_factors[2:end]


    if ismissing(factor)
        if sum(significativeFactors) == 0
            error("Please pass a factor for the residuals groups. There aren't significative factors in the model.")
        else
            i = findfirst(significativeFactors)
            factor = mod.names[i+1]
        end
    end


    boxplot(df[:,factor],residuals(mod),legend = false,
        color = :blueviolet)
    xlabel!(string(factor))
    title!("Residuals Homocedasticity")
end
    


function qqplotResiduals(mod::Model)
    qqnorm(residuals(mod))
    xlabel!("Theorical Quantiles")
    ylabel!("Sample Quantiles")
end
