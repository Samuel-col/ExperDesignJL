
using LinearAlgebra, Distributions, DataFrames

import Base: summary
import LinearAlgebra: rank

struct Factor # Representa factores
    val::Array{Int64,1} # Valores del vector
    lev::Array{Any,1} # Los niveles que representan los valores asignados

    # Método Constructivo
    function Factor(x)
        levels = x |> unique |> sort #  Vector de niveles
        convert = Dict{Any,Int64}() # Diccionario de traducción
        i = 0
        for l in levels
            convert[l] = i
            i += 1
        end
        formated = x |> length |> zeros .|> Int
        for i in 1:length(x)
            formated[i] = convert[x[i]]
        end

        new(formated,levels) # Creación del objeto
    end
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
                tmp = ones(n,1)
                fac = Factor(data[:,names[i]])
                for lev in 1:length(fac.lev)
                    X = hcat(X,fac.val .== lev) # Creación de la matriz diseño
                    tmp = hcat(tmp,fac.val .== lev)
                end
                push!(Xi,tmp[:,2:end])
            end
        end
        y = data[:,names[1]]
        new(y,X,Xi,names)
    end
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
    gl = []
    for i = 2:length(mod.Xi) # Grados de libertad de la regresión
        if rank(mod.Xi[i])!=1
            push!(gl,rank(mod.Xi[i])-1)
        else
            push!(gl,1)
        end
    end
    push!(gl,length(mod.y)-rank(mod.X)) # Grados de libertad de los residuales
    push!(gl,length(mod.y)-1) # Grados de libertad totales
    return tuple(gl...)
end

function CM1(mod::Model)
    cm = []
    sc = SC1(mod)
    gl = gl1(mod)
    for i in 1:length(sc)
        push!(cm,sc[i]/gl[i])
    end
    return tuple(cm...)
end

function FStat1(mod::Model)
    fStat = []
    cm = CM1(mod)
    for i = 1:(length(cm)-2)
        push!(fStat,cm[i]/cm[end-1])
    end
    return tuple(fStat...)
end

function pValue1(mod::Model)
    pv = []
    gl = gl1(mod)
    fStats = FStat1(mod)
    for i = 1:length(fStats)
        push!(pv, ccdf(FDist(gl[i],gl[end-1]),fStats[i]))
    end
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
    gl = []
    for i = 2:length(mod.Xi) # Grados de libertad de la regresión
        if rank(mod.Xi[i])!=1
            push!(gl,rank(mod.Xi[i])-1)
        else
            push!(gl,1)
        end
    end
    push!(gl,length(mod.y)-rank(mod.X)) # Grados de libertad de los residuales
    return tuple(gl...)
end

function CM2(mod::Model)
    cm = []
    sc = SC2(mod)
    gl = gl2(mod)
    for i in 1:length(sc)
        push!(cm,sc[i]/gl[i])
    end
    return tuple(cm...)
end

function FStat2(mod::Model)
    fStat = []
    cm = CM2(mod)
    σ = CM1(mod)[end-1]
    for λ in cm
        push!(fStat,λ/σ)
    end
    return tuple(fStat...)
end

function pValue2(mod::Model)
    pv = []
    gl = gl2(mod)
    fStats = FStat2(mod)
    for i in 1:length(fStats)
        push!(pv,ccdf(FDist(gl[i],gl[end]),fStats[i]))
    end
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

function summary(mod::Model)
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
    return (σ=σ,R²=R,R²adj=R_adj,Fstat=F,pValue=pVal,N_Obs=n,N_Var=k,regDF=gl)
end
