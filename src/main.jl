module ExprDesign

    export Factor, Model, summary, display
    export residuals, YEst, θ, EstimableBase
    export anova1, regAn, anova2
    export plotResiduals, qqplotResiduals
    
    include("objetos.jl")

    include("base_functions.jl")

    include("anova_type1.jl")

    include("anova_type2.jl")

    include("graphics.jl")

end