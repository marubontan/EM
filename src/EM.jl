using Distributions

struct EMResults{T<:Array}
    mu::T
    sigma::T
    mix::T
    posterior::T
    logLikeLihoods::T
    iterCount::Int
end


struct Initialization
    mu::Array
    sigma::Array
    mix::Array
end


# TODO: there are two types of bugs. One, the sigma becomes non-positive definite. Two, the whole posteriors become zero and parameters become NaN.
function EM(data, k; initialization=nothing)

    if initialization == nothing
        mu, sigma, mix = initializeParameters(data, k)
    elseif typeof(initialization) == Initialization
        mu, sigma, mix = (initialization.mu, initialization.sigma, initialization.mix)
    else
        error("The argument, initialization, is not valid.")
    end

    posterior = eStep(data, mu, sigma, mix)
    muArray = []
    sigmaArray = []
    mixArray = []
    posteriorArray = []
    logLikelihoods = []
    iterCount = 0
    while iterCount >= 2 && !checkConvergence(logLikelihoods[end-1], logLikelihoods[end])

        mu, sigma, mix = mStep(data, posterior)
        # TODO: do proper experiment to check the case this needs
        sigma = [adjustToSymmetricMatrix(sig) for sig in sigma]
        posteriorTemp = eStep(data, mu, sigma, mix)

        push!(logLikelihoods, calcLogLikelihood(data, mu, sigma, mix, posteriorTemp))
        push!(muArray, mu)
        push!(sigmaArray, sigma)
        push!(mixArray, mix)
        push!(posteriorArray, posteriorTemp)

        iterCount += 1
        posterior = posteriorTemp
    end
    return EMResults(muArray, sigmaArray, mixArray, posteriorArray, logLikelihoods, iterCount)
end

function initializeParameters(data, k::Int)
    numberOfVariables = size(data)[2]
    mu = [rand(Normal(0, 100), numberOfVariables) for _ in 1:k]
    sigma = [10000 * eye(numberOfVariables) for _ in 1:k]
    mixTemp = rand(k)
    mix = mixTemp / sum(mixTemp)
    return mu, sigma, mix
end

function eStep(data::Array, mu::Array, sigma::Array, mix::Array)
    posteriorArray = []
    for i in 1:size(data)[1]
        posteriors = Array{Float64}(length(mix))
        for j in 1:length(mix)
            posteriors[j] = calculatePosterior(data[i,:], mu[j], sigma[j], mix[j])
        end
        push!(posteriorArray, makeArrayRatio(posteriors))
    end
    return posteriorArray
end

function mStep(data, posteriors)
    numberOfClusterDataPoints = estimateNumberOfClusterDataPoints(posteriors)

    updatedMu = updateMu(posteriors, data, numberOfClusterDataPoints)
    updatedSigma = updateSigma(posteriors, data, numberOfClusterDataPoints,updatedMu)
    updatedMix = updateMix(numberOfClusterDataPoints)

    return updatedMu, updatedSigma, updatedMix
end

function checkConvergence(posterior, updatedPosterior)
    return isapprox(posterior, updatedPosterior)
end

function calcLogLikelihood(data, mu, sigma, mix, posterior)
    logLikelihood = 0.0
    for i in 1:size(data)[1]
        for k in 1:length(mix)
            logLikelihood += posterior[i][k] * (log(mix[k]) - (length(mu[1])/2) * log(2 * mix[k]) + (1/2) * log(1/det(sigma[k])) - (1/2) * (data[i] - mu[k])' * inv(sigma[k]) * (data[i] - mu[k]))
        end
    end
    return logLikelihood
end

function calculatePosterior(data::Array, mu::Array, sigma::Array, prior::Float64)
    return prior * pdf(MvNormal(mu, sigma), data)
end

function makeArrayRatio(posteriors)
    return sum(posteriors) == 0 ? posteriors : posteriors / sum(posteriors)
end

function estimateNumberOfClusterDataPoints(posteriors::Array)
    clusterNum = length(posteriors[1])
    numberOfClusterDataPoints = zeros(clusterNum)
    for posterior in posteriors
        numberOfClusterDataPoints += posterior
    end
    return numberOfClusterDataPoints
end

function updateMu(posteriors, data, numberOfClusterDataPoints)
    updatedMuArray = []
    for k in 1:length(numberOfClusterDataPoints)
        muSum = 0
        for i in 1:size(data)[1]
            muSum += posteriors[i][k] * data[i, :]
        end
        push!(updatedMuArray, muSum/numberOfClusterDataPoints[k])
    end
    return updatedMuArray
end

function updateSigma(posteriors, data, numberOfClusterDataPoints, mu)
    updatedSigmaArray = []
    for k in 1:length(numberOfClusterDataPoints)
        sigmaSum = 0
        for i in 1:size(data)[1]
            sigmaSum += posteriors[i][k] * (data[i, :] - mu[k]) * (data[i, :] - mu[k])'
        end
        push!(updatedSigmaArray, sigmaSum/numberOfClusterDataPoints[k])
    end
    return updatedSigmaArray
end

function updateMix(numberOfClusterDataPoints)
    return numberOfClusterDataPoints / sum(numberOfClusterDataPoints)
end

function adjustToSymmetricMatrix(matrix)
    for r in 1:size(matrix)[1]
        for c in 1:size(matrix)[2]
            if c > r
                matrix[r, c] = matrix[c, r]
            end
        end
    end
    return matrix
end
