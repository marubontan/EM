using Distributions
include("../src/EM.jl")

function makeData()
    groupOne = rand(MvNormal([10.0, 10.0], 5.0 * eye(2)), 1000)
    groupTwo = rand(MvNormal([0.0, 0.0], 10 * eye(2)), 1000)
    groupThree = rand(MvNormal([100.0, 100.0], 20 * eye(2)), 1000)
    groupFour = rand(MvNormal([100.0, 50.0], 20 * eye(2)), 1000)
    return hcat(groupOne, groupTwo, groupThree, groupFour)'
end

srand(1234)
trainData = makeData()

timeEM = @elapsed resultEM = EM(trainData, 4)

# time
@show timeEM
