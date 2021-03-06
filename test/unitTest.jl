using Base.Test
using Distributions
include("../src/EM.jl")

@testset "support function" begin
    sigmaA = eye(2)
    sigmaA[1,2] = 3.0
    @test adjustToSymmetricMatrix(sigmaA) == eye(2)
    sigmaB = eye(5)
    sigmaB[1,2] = 8.0
    @test adjustToSymmetricMatrix(sigmaB) == eye(5)

end

data = [1.0 2.0; 2.0 1.0; 100.0 100.0; 90.0 110.0]
mu = [[0.0, 0.0], [100.0, 100.0]]
sigma = [[1.0 0.0; 0.0 1.0], [2.0 0.0; 0.0 2.0]]
mix = [0.3, 0.7]
@testset "eStep" begin
    @test calculatePosterior(data[1, :], mu[1], sigma[1], mix[1]) == 0.3 * pdf(MvNormal([0.0, 0.0], [1 0; 0 1]), [1, 2])
    posteriors = [1.0, 2.0, 3.0, 4.0]
    @test makeArrayRatio(posteriors) == [0.1, 0.2, 0.3, 0.4]

    calculatedPosterior = eStep(data, mu, sigma, mix)
    @test length(calculatedPosterior) == size(data)[1]
    @test length(calculatedPosterior[1]) == length(mix)
end

@testset "mStep" begin
    dataA = [1.0 2.0; 100.0 100.0]
    dataB = [1.0 5.0; 1000.0 1000.0]
    mu = [[1.1, 2.1], [100.1, 104.6]]
    posteriors = [[1.0, 0.0], [0.0, 1.0]]
    @test estimateNumberOfClusterDataPoints(posteriors) == [1.0, 1.0]

    estimatedNumberOfClusterDataPoints = [1.0, 1.0]
    @test updateMu(posteriors, dataA, estimatedNumberOfClusterDataPoints) == [(posteriors[1][1] * dataA[1, :] + posteriors[2][1] * dataA[2, :]) / estimatedNumberOfClusterDataPoints[1],
    (posteriors[1][2] * dataA[1, :] + posteriors[2][2] * dataA[2, :]) / estimatedNumberOfClusterDataPoints[2]]
    @test updateMu(posteriors, dataB, estimatedNumberOfClusterDataPoints) == [(posteriors[1][1] * dataB[1, :] + posteriors[2][1] * dataB[2, :]) / estimatedNumberOfClusterDataPoints[1],
    (posteriors[1][2] * dataB[1, :] + posteriors[2][2] * dataB[2, :]) / estimatedNumberOfClusterDataPoints[2]]

    @test updateSigma(posteriors, dataA, estimatedNumberOfClusterDataPoints, mu) == [(posteriors[1][1] * (dataA[1, :] - mu[1]) * (dataA[1, :] - mu[1])' + posteriors[2][1] * (dataA[2, :] - mu[1]) * (dataA[2, :] - mu[1])') / estimatedNumberOfClusterDataPoints[1],
    (posteriors[1][2] * (dataA[1, :] - mu[2]) * (dataA[1, :] - mu[2])' + posteriors[2][2] * (dataA[2, :] - mu[2]) * (dataA[2, :] - mu[2])') / estimatedNumberOfClusterDataPoints[2]]
    @test updateSigma(posteriors, dataB, estimatedNumberOfClusterDataPoints, mu) == [(posteriors[1][1] * (dataB[1, :] - mu[1]) * (dataB[1, :] - mu[1])' + posteriors[2][1] * (dataB[2, :] - mu[1]) * (dataB[2, :] - mu[1])') / estimatedNumberOfClusterDataPoints[1],
    (posteriors[1][2] * (dataB[1, :] - mu[2]) * (dataB[1, :] - mu[2])' + posteriors[2][2] * (dataB[2, :] - mu[2]) * (dataB[2, :] - mu[2])') / estimatedNumberOfClusterDataPoints[2]]

    @test updateMix(estimatedNumberOfClusterDataPoints) == [0.5, 0.5]
    updatedMu, updatedSigma, updatedMix = mStep(dataA, posteriors)
    @test isa(updatedMu, Array)
    @test isa(updatedSigma, Array)
    @test isa(updatedMix, Array)
    @test length(updatedMu) == length(posteriors[1])
    @test length(updatedSigma) == length(posteriors[1])
    @test length(updatedMix) == length(posteriors[1])
end

@testset "EM" begin
    groupOneA = rand(MvNormal([1,1], eye(2)), 100)
    groupTwoA = rand(MvNormal([10,10], eye(2)), 100)
    dataA = hcat(groupOneA, groupTwoA)'

    muInit = [[0.0, 0.0], [20.0, 20.0]]
    sigmaInit = [10.0 * eye(2), 10.0 * eye(2)]
    mixInit = [0.5, 0.5]
    @test_nowarn EM(dataA, 2)
    @test_nowarn EM(dataA, 2; initialization=Initialization(muInit, sigmaInit, mixInit))

    muInitError = [[0.0, 0.0], [20.0, 20.0], [1.0, 2.0]]
    @test_throws AssertionError EM(dataA, 2; initialization=Initialization(muInitError, sigmaInit, mixInit))

    groupOneB = rand(MvNormal([1,1], eye(2)), 100)
    groupTwoB = rand(MvNormal([1000,1000], eye(2)), 1000)
    dataB = hcat(groupOneB, groupTwoB)'
    @test_nowarn EM(dataB, 2)

    groupOneC = rand(MvNormal([1,1,1], eye(3)), 100)
    groupTwoC = rand(MvNormal([1000,1000,10], eye(3)), 1000)
    dataC = hcat(groupOneC, groupTwoC)'
    @test_nowarn EM(dataC, 2)

    groupOneD = rand(MvNormal([1,1,1,4], eye(4)), 100)
    groupTwoD = rand(MvNormal([1000,1000,10,40], eye(4)), 1000)
    groupThreeD = rand(MvNormal([-1000,-1000,10,50], eye(4)), 1000)
    dataD = hcat(groupOneD, groupTwoD, groupThreeD)'
    @test_nowarn EM(dataD, 3)

    @test_nowarn EM(dataA, 2; maxIter=100) 

    @test_throws AssertionError EM(dataA, 1000)
end

@testset "checkConvergence" begin
    @test checkConvergence(1, 1)
end

@testset "logLikelihood" begin
    dataA = [1.0 1.0; 10.0 10.0]
    muA = [[2.0, 2.0], [7.0, 7.0]]
    sigmaA = [[1.0 0.0; 0.0 1.0], [2.0 0.0; 0.0 2.0]]
    mixA = [0.3, 0.7]
    posteriorA = [[0.8, 0.2], [0.2, 0.8]]
    @test_nowarn calcLogLikelihood(dataA, muA, sigmaA, mixA, posteriorA)
end
