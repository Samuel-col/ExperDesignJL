# Funciones elementales para el paquete

import LinearAlgebra: rank

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

function rank(V::Array{<:Real,1})
    return 1
end

function removeChar(c::Char,s::String)::String
    l = length(s)
    ind = findall(c,s)
    newInd = symdiff(1:l,ind)
    return s[newInd]
end

function stringBreak(s::String,indices::Vector{Int})::Vector{String}
    push!(indices,0)
    push!(indices,length(s)+1)
    indices = sort(indices)
    return [s[(indices[i-1]+1):(indices[i]-1)] for i in 2:length(indices)]
end

function readFormula(s::String)
    specialChar = ['~','+','*',':']
    
    s = removeChar(' ',s)
    l = length(s)
    equal_position = findfirst('~',s)
    
    y_str = SubString(s,1,equal_position-1)
    x_str = SubString(s,equal_position+1,l)
    
    #specialChar_pos = vcat(findall.(specialChar,x_str)...)
    plus_pos = findall('+',x_str)
    regressors_str = stringBreak(string(x_str),plus_pos)

    for i in 1:length(regressors_str)
        fctr = regressors_str[i]
        if occursin('*',fctr)
            interaction = replace(fctr,'*' => ':')
            regressors_str[i] = interaction
            actors = stringBreak(fctr,[findfirst('*',fctr)])
            push!(regressors_str,actors...)
        end
    end
    
    return y_str,sort(regressors_str,by = length)
end


