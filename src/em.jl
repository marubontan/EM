using Distributions

function EM(data, k)
    mu = [rand(Normal(0, 100), 2) for i in 1:3]
    sigma = rand(Wishart(3, 1000 * eye(2)), 3)
    mixTemp = rand(3)
    mix = mixTemp / sum(mixTemp)

    posterior = eStep(data, mu, sigma, mix)
    while true
        mu, sigma, mix = mStep(data, posterior)
        sigma = [adjustToSymmetricMatrix(sig) for sig in sigma]
        posteriorTemp = eStep(data, mu, sigma, mix)
        if isapprox(posterior, posteriorTemp)
            break
        end
        posterior = posteriorTemp
    end
    return mu, sigma, mix
end

# TODO: refactoring
function eStep(data::Array, mu::Array, sigma::Array, mix::Array)
    posteriorArray = []
    for i in 1:size(data)[1]
        posteriors = Array{Float64}(length(mix))
        for j in 1:length(mix)
            posterior = getPosterior(data[i,:], mu[j], sigma[j], mix[j])
            posteriors[j] = posterior
        end
        push!(posteriorArray, getPosteriorProbability(posteriors))
    end
    return posteriorArray
end

function getPosterior(data::Array, mu::Array, sigma::Array, mix::Float64)
    return mix * pdf(MvNormal(mu, sigma), data)
end

function getPosteriorProbability(posteriors)
    return sum(posteriors) == 0 ? posteriors : posteriors / sum(posteriors)
end

function adjustToSymmetricMatrix(matrix)
    for r in 1:size(matrix)[1]
        for c in 1:size(matrix)[2]
            if c + r - size(matrix)[1] > 0
                matrix[r, c] = matrix[c, r]
            end
        end
    end
    return matrix
end

function mStep(data, posteriors)
    # TODO: fix dirty variable name
    estimatedNumberOfClusterDataPoints = estimateNumberOfClusterDataPoints(posteriors)

    updatedMu = updateMu(posteriors, data, estimatedNumberOfClusterDataPoints)
    updatedSigma = updateSigma(posteriors, data, estimatedNumberOfClusterDataPoints,updatedMu)
    updatedMix = updateMix(estimatedNumberOfClusterDataPoints)

    return updatedMu, updatedSigma, updatedMix
end

function estimateNumberOfClusterDataPoints(posteriors::Array)
    clusterNum = length(posteriors[1])
    numberOfClusterDataPoints = zeros(clusterNum)
    for posterior in posteriors
        numberOfClusterDataPoints += posterior
    end
    return numberOfClusterDataPoints
end

function updateMu(posteriors, data, estimatedNumberOfClusterDataPoints)
    updatedMuArray = []
    for k in 1:length(estimatedNumberOfClusterDataPoints)
        muSum = 0
        for i in 1:size(data)[1]
            muSum += posteriors[i][k] * data[i, :]
        end
        push!(updatedMuArray, muSum/estimatedNumberOfClusterDataPoints[k])
    end
    return updatedMuArray
end

function updateSigma(posteriors, data, estimatedNumberOfClusterDataPoints, mu)
    updatedSigmaArray = []
    for k in 1:length(estimatedNumberOfClusterDataPoints)
        sigmaSum = 0
        for i in 1:size(data)[1]
            sigmaSum += posteriors[i][k] * (data[i, :] - mu[k]) * (data[i, :] - mu[k])'
        end
        push!(updatedSigmaArray, sigmaSum/estimatedNumberOfClusterDataPoints[k])
    end
    return updatedSigmaArray
end

function updateMix(estimatedNumberOfClusterDataPoints)
    return estimatedNumberOfClusterDataPoints / sum(estimatedNumberOfClusterDataPoints)
end
