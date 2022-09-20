# Gráfico asociados al paquete
using StatsPlots


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
