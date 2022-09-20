# ANOVA tipo I

using Distributions

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
