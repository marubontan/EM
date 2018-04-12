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

function eStep(data, mu, sigma, mix)
    dataNum = size(data)[1]
    K = length(mix)

    posteriors = []
    for i in 1:dataNum
        posteriorK = []

        for k in 1:K
            try
                #sigma[k] = adjustToSymmetricMatrix(sigma[k])
                push!(posteriorK, mix[k] * pdf(MvNormal(mu[k], sigma[k]), data[i, :]))
            catch
                println(sigma[k])
            end
        end
        all = sum(posteriorK)
        if all != 0
            posteriorK = posteriorK / all
        end
        push!(posteriors, posteriorK)
    end
    return posteriors
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
