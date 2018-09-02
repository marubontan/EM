using Distributions

struct EMResults
    mu::Array{Array{Array{Float64, 1}}}
    sigma::Array{Array{Array{Float64, 2}}}
    mix::Array{Array{Float64, 1}}
    posterior::Array{Array{Array{Float64, 1}}}
    logLikelihoods::Array{Float64}
    iterCount::Int
    maxIter::Int
    converged::Bool
end


struct Initialization
    mu::Array{Array{Float64, 1}}
    sigma::Array{Array{Float64, 2}}
    mix::Array{Float64, 1}
end


# TODO: there are two types of bugs. One, the sigma becomes non-positive definite. Two, the whole posteriors become zero and parameters become NaN.
function EM(data::Array{Float64, 2}, k::Int; initialization=nothing, maxIter=10000)

    rowSize, colSize = size(data)

    @assert 2 <= k < rowSize

    if initialization == nothing

        mu, sigma, mix = initializeParameters(data, k)
    elseif typeof(initialization) == Initialization

        @assert length(initialization.mu) == k
        @assert length(initialization.mu[1]) == colSize

        @assert length(initialization.sigma) == k
        @assert size(initialization.sigma[1]) == (colSize, colSize)

        @assert length(initialization.mix) == k
        @assert length(initialization.mu[1]) == colSize

        mu, sigma, mix = (initialization.mu, initialization.sigma, initialization.mix)
    else
        error("The argument, initialization, is not valid.")
    end

    posterior = eStep(data, mu, sigma, mix)
    muArray = Array{Array{Array{Float64, 1}}}(undef, maxIter)
    sigmaArray = Array{Array{Array{Float64, 2}}}(undef, maxIter)
    mixArray = Array{Array{Float64, 1}}(undef, maxIter)
    posteriorArray = Array{Array{Array{Float64, 1}}}(undef, maxIter)
    logLikelihoods = Array{Float64}(undef, maxIter)
    iterCount = zero(1)
    converged = false
    while iterCount < maxIter

        mu, sigma, mix = mStep(data, posterior)
        # TODO: do proper experiment to check the case this needs
        sigma = [adjustToSymmetricMatrix(sig) for sig in sigma]
        posterior = eStep(data, mu, sigma, mix)

        logLikelihoods[iterCount+1] = calcLogLikelihood(data, mu, sigma, mix, posterior)
        muArray[iterCount+1] = mu
        sigmaArray[iterCount+1] = sigma
        mixArray[iterCount+1] = mix
        posteriorArray[iterCount+1] = posterior
        iterCount += 1
        if iterCount >= 2
            if checkConvergence(logLikelihoods[iterCount-1], logLikelihoods[iterCount])
                converged = true
                break
            end
        end
    end

    return EMResults(muArray[1:iterCount],
                     sigmaArray[1:iterCount],
                     mixArray[1:iterCount],
                     posteriorArray[1:iterCount],
                     logLikelihoods[1:iterCount],
                     iterCount,
                     maxIter,
                     converged)
end


#TODO: inappropriate initialization
function initializeParameters(data, k::Int)

    numberOfVariables = size(data)[2]
    mu = [rand(Normal(0, 100), numberOfVariables) for _ in 1:k]
    sigma = [10000 * make_eye(numberOfVariables) for _ in 1:k]
    mixTemp = rand(k)
    mix = mixTemp / sum(mixTemp)
    return mu, sigma, mix
end


function eStep(data::Array{Float64, 2},
               mu::Array{Array{Float64, 1}},
               sigma::Array{Array{Float64, 2}},
               mix::Array{Float64, 1})

    numberOfDataPoint = size(data)[1]

    posteriorArray = Array{Array{Float64, 1}}(undef, numberOfDataPoint)
    for dataIndex in 1:numberOfDataPoint
        posteriors = Array{Float64}(undef, length(mix))
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

#TODO: should be more strict??
function checkConvergence(posterior, updatedPosterior)

    return isapprox(posterior, updatedPosterior)
end


function calcLogLikelihood(data, mu, sigma, mix, posterior)

    logLikelihood = zero(Float64)
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


function estimateNumberOfClusterDataPoints(posteriors::Array{Array{Float64, 1}, 1})

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

    updatedMuArray = Array{Array{Float64, 1}}(undef, numberOfCluster)
    for cluster in 1:numberOfCluster
        muSum = zero(Float64)
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

    rowSize, colSize = size(data)
    numberOfCluster = length(numberOfClusterDataPoints)

    updatedSigmaArray = Array{Array{Float64, 2}}(undef, numberOfCluster)
    for cluster in 1:numberOfCluster
        sigmaSum = zero(rand(colSize, colSize))
        for dataIndex in 1:rowSize
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

function make_eye(num)
    eye = zeros(num, num)
    for i = 1:num
        eye[i, i] = 1.0
    end
    return eye
end
