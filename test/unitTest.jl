using Base.Test
using Distributions

include("../src/em.jl")
@testset "support function" begin
sigma = eye(2)
sigma[1,2] = 3

@test adjustToSymmetricMatrix(sigma) == [1 0;0 1]
end

@testset "E and M step" begin
# data
groupOne = rand(MvNormal([1,1], eye(2)), 100)
groupTwo = rand(MvNormal([10,10], eye(2)), 100)
groupThree = rand(MvNormal([100, 100], eye(2)), 100)

data = hcat(groupOne, groupTwo, groupThree)'

mu = [rand(Normal(0, 10), 2) for i in 1:3]
sigma = rand(Wishart(4, eye(2)), 3)
mixTemp = rand(3)
mix = mixTemp / sum(mixTemp)

posteriors = eStep(data, mu, sigma, mix)
muTemp, sigmaTemp, mixTemp = mStep(data, posteriors)
end

@testset "EM" begin
# data
groupOne = rand(MvNormal([1,1], eye(2)), 100)
groupTwo = rand(MvNormal([10,10], eye(2)), 100)
groupThree = rand(MvNormal([100, 100], eye(2)), 100)

data = hcat(groupOne, groupTwo, groupThree)'

println(EM(data, 3))
end
