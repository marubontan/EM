include("../src/EM.jl")
using Distributions

groupOne = rand(MvNormal([1,1], eye(2)), 100)
groupTwo = rand(MvNormal([10,10], eye(2)), 100)
groupThree = rand(MvNormal([100,100], eye(2)), 100)

data = hcat(groupOne, groupTwo, groupThree)'

emResults = EM(data, 3)
println("mean: [1, 1],[10, 10], [100, 100]")
@show emResults.mu[end]
println("sigma: [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]")
@show emResults.sigma[end]
println("mix: [0.3, 0.3, 0.3]")
@show emResults.mix[end]
@show emResults.logLikelihoods[end]
@show emResults.iterCount
