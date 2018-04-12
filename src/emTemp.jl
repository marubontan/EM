using Distributions

function EM(data, k::Int)
    paramPi = ones(k) / k
    u = rand(k, size(data)[2])
    sigma = [eye(size(data)[2]) for i in 1:k]
    posterior = eStep(data, u, paramPi, sigma)
    while true
        u, sigma, paramPi = mStep(data, posterior, k)
        println("u:",u)
        posteriorTemp = eStep(data, u, paramPi, sigma)
        if isapprox(posterior, posteriorTemp)
            break
        else
            posterior = posteriorTemp
        end
    end
    return u, sigma, paramPi
end

function eStep(x, paramU::Array, paramPi::Array, paramSigma::Array)
    posteriors = []
    for i in 1:size(x)[1]
        posteriorD = Array{Float64}(length(paramPi))
        dataPoint = x[i, :]
        for k in 1:length(paramPi)
            uK = Array(paramU[k, :])
            sigmaK = paramSigma[k, :][1]
            sigmaK[1,2] = sigmaK[2,1]
            posteriorK = paramPi[k] * pdf(MvNormal(uK, sigmaK), dataPoint)
            posteriorD[k] = posteriorK
        end
        push!(posteriors, posteriorD)
    end
    return posteriors
end

function mStep(data, posteriors, K)
    u = zeros(K, size(data)[2])
    sigma = []
    paramPi = []
    for k in 1:K
        nK = 0
        uKSum = zeros(size(data)[2])
        for i in 1:size(data)[1]
            nK += posteriors[i][k]
            uKSum += posteriors[i][k] * data[i, :]
        end
        uK = uKSum / nK
        sigmaKSum = zeros(size(data)[2], size(data)[2])
        for i in 1:size(data)[1]
            sigmaKSum += posteriors[i][k] * (data[i, :] - nK) * (data[i, :] - nK)'
        end
        sigmaK = sigmaKSum / nK
        piK = nK / size(data)[1]

        u[k, :] = uK
        push!(sigma, sigmaK)
        push!(paramPi, piK)
    end
    return u, sigma, paramPi
end
