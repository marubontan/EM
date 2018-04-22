include("../src/em.jl")
using Distributions

groupOne = rand(MvNormal([1,1], eye(2)), 100)
groupTwo = rand(MvNormal([10,10], eye(2)), 100)
groupThree = rand(MvNormal([100,100], eye(2)), 100)

data = hcat(groupOne, groupTwo, groupThree)'

println(EM(data, 3))

