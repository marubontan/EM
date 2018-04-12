include("./src/em.jl")
using Distributions

groupOne = rand(MvNormal([1,1], eye(2)), 100)
groupTwo = rand(MvNormal([10,10], eye(2)), 100)

data = hcat(groupOne, groupTwo)'

println(EM(data, 2))
