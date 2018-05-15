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
    converged = false
    while !converged

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
        if iterCount >= 2
            if checkConvergence(logLikelihoods[end-1], logLikelihoods[end])
                converged = true
            end
        end
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


function eStep(data::Array{Float64, 2},
               mu::Array{Array{Float64, 1}},
               sigma::Array{Array{Float64, 2}},
               mix::Array{Float64, 1})

    numberOfDataPoint = size(data)[1]

    posteriorArray = Array{Array{Float64, 1}}(numberOfDataPoint)
    for dataIndex in 1:numberOfDataPoint
        posteriors = Array{Float64}(length(mix))
        for j in 1:length(mix)
            posteriors[j] = calculatePosterior(data[dataIndex,:],
                                               mu[j],
                                               sigma[j],
                                               mix[j])
        end
        posteriorArray[dataIndex] = makeArrayRatio(posteriors)
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


function calculatePosterior(data::Array{Float64, 1},
                            mu::Array{Float64, 1},
                            sigma::Array{Float64, 2},
                            prior::Float64)

    return prior * pdf(MvNormal(mu, sigma), data)
end


function makeArrayRatio(posteriors::Array{Float64, 1})

    return sum(posteriors) == 0.0 ? posteriors : posteriors / sum(posteriors)
end


function estimateNumberOfClusterDataPoints(posteriors::Array{Float64})

    clusterNum = length(posteriors[1])
    numberOfClusterDataPoints = zeros(clusterNum)
    for posterior in posteriors
        numberOfClusterDataPoints += posterior
    end

    return numberOfClusterDataPoints
end


function updateMu(posteriors::Array{Array{Float64, 1}},
                  data::Array{Float64, 2},
                  numberOfClusterDataPoints::Array{Float64})

    numberOfCluster = length(numberOfClusterDataPoints)

    updatedMuArray = Array{Array{Float64, 1}}(numberOfCluster)
    for cluster in 1:numberOfCluster
        muSum = 0
        for dataIndex in 1:size(data)[1]
            muSum += posteriors[dataIndex][cluster] * data[dataIndex, :]
        end
        updatedMuArray[cluster] = muSum/numberOfClusterDataPoints[cluster]
    end
    return updatedMuArray
end


function updateSigma(posteriors::Array{Array{Float64, 1}},

    data::Array{Float64, 2},
    numberOfClusterDataPoints::Array{Float64},
    mu::Array{Array{Float64, 1}})

    numberOfCluster = length(numberOfClusterDataPoints)

    updatedSigmaArray = Array{Array{Float64, 2}}(numberOfCluster)
    for cluster in 1:numberOfCluster
        sigmaSum = 0
        for dataIndex in 1:size(data)[1]
            sigmaSum += posteriors[dataIndex][cluster] * (data[dataIndex, :] - mu[cluster]) * (data[dataIndex, :] - mu[cluster])'
        end
        updatedSigmaArray[cluster] = sigmaSum/numberOfClusterDataPoints[cluster]
    end
    return updatedSigmaArray
end


function updateMix(numberOfClusterDataPoints::Array{Float64})

    return numberOfClusterDataPoints / sum(numberOfClusterDataPoints)
end


function adjustToSymmetricMatrix(matrix::Array{Float64, 2})

    rowSize, colSize = size(matrix)

    symmetricMatrix = deepcopy(matrix)

    for rowIndex in 1:rowSize
        for colIndex in 1:colSize
            if colIndex > rowIndex
                symmetricMatrix[rowIndex, colIndex] = matrix[colIndex, rowIndex]
            end
        end
    end
    return symmetricMatrix
end
