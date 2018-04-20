using Distributions

function EM(data, k)
    mu = [rand(Normal(0, 100), 2) for i in 1:3]
    sigma = rand(Wishart(3, eye(2)), 3)
    mixTemp = rand(3)
    mix = mixTemp / sum(mixTemp)

    posterior = eStep(data, mu, sigma, mix)
    while true
        mu, sigma, mix = mStep(data, posterior)
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
    return posteriors / sum(posteriors)
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
    clusterNumber = length(posteriors[1])
    dataCharacteristics = size(data)[2]

    nkArray = []
    for (index,posterior) in enumerate(posteriors)
        if index == 1
            nkArray = posterior
        else
            nkArray += posterior
        end
    end
    ukArray = []
    sigmaArray = []
    for (index, nK) in enumerate(nkArray)
        uK = zeros(size(data)[2])
        for i in 1:size(data)[1]
            uK += posteriors[i][index] * data[i, :]
        end
        uK /= nK

        sigma = zeros(dataCharacteristics, dataCharacteristics)
        for i in 1:size(data)[1]
            sigma += posteriors[i][index] * (data[i, :] - uK) * (data[i, :] - uK)'
        end
        sigma /= nK

        push!(ukArray, uK)
        push!(sigmaArray, sigma)
    end
    mix = nkArray / sum(nkArray)
    return ukArray, sigmaArray, mix
end

function estimateNumberOfClusterDataPoints(posteriors::Array)
    clusterNum = length(posteriors[1])
    numberOfClusterDataPoints = zeros(clusterNum)
    for posterior in posteriors
        numberOfClusterDataPoints += posterior
    end
    return numberOfClusterDataPoints
end
