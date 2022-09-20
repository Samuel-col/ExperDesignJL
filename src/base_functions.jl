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

