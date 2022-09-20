# ANOVA tipo II

using Distributions

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