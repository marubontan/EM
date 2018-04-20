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
    posteriors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    @test estimateNumberOfClusterDataPoints(posteriors) == [2.0, 2.0]
    @test updateMu
    #@test updateSigma
    #@test updateMix
end
