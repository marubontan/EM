using Base.Test
using Distributions

include("../src/em.jl")
@testset "support function" begin
    sigma = eye(2)
    sigma[1,2] = 3

    @test adjustToSymmetricMatrix(sigma) == [1 0;0 1]
end

data = [1.0 2.0; 2.0 1.0; 100.0 100.0; 90.0 110.0]
mu = [[0.0, 0.0], [100.0, 100.0]]
sigma = [[1.0 0.0; 0.0 1.0], [2.0 0.0; 0.0 2.0]]
mix = [0.3, 0.7]
@testset "eStep" begin
    @test getPosterior(data[1, :], mu[1], sigma[1], mix[1]) == 0.3 * pdf(MvNormal([0.0, 0.0], [1 0; 0 1]), [1, 2])
    posteriors = [1, 2, 3, 4]
    @test getPosteriorProbability(posteriors) == [0.1, 0.2, 0.3, 0.4]
    # TODO: write proper test
    println(eStep(data, mu, sigma, mix))
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
    println(mStep(dataA, posteriors))
end
