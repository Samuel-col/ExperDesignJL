# Objetos básicos del paquete

using LinearAlgebra, Distributions, DataFrames

import Base: summary
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


