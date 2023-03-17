using Distributions


struct TruncatedNormal <: Gen.Distribution{Float64} end
const truncated_normal = TruncatedNormal()


function Gen.random(::TruncatedNormal, mu::Real, std::Real, boundary::Vector{Float64})
    dist = Distributions.TruncatedNormal(mu, std, boundary[1], boundary[2])
    rand(dist)
end


function Gen.logpdf(::TruncatedNormal, x::Real, mu::Real, std::Real, boundary::Vector{Float64})
    dist = Distributions.TruncatedNormal(mu, std, boundary[1], boundary[2])
    Distributions.logpdf(dist, x)
end


(::TruncatedNormal)(mu, std, boundary) = Gen.random(truncated_normal, mu, std, boundary)


Gen.has_output_grad(::TruncatedNormal) = false
Gen.has_argument_grads(::TruncatedNormal) = (false, false, false)
Gen.logpdf_grad(::TruncatedNormal, value::Set, args...) = (nothing,)


export truncated_normal