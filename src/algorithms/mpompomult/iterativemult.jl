

struct MPOMPOIterativeMultCache{_MPO, _IMPO, _OMPO, _H} <: AbstractMPOMPOMultCache
	mpo::_MPO
	impo::_IMPO
	ompo::_OMPO
	hstorage::_H
end

scalar_type(m::MPOMPOIterativeMultCache) = scalar_type(m.ompo)

sweep!(m::MPOMPOIterativeMultCache, alg::AbstractMPSArith, workspace = scalar_type(m)[]) = iterative_error_2(
    vcat(_leftsweep!(m, alg, workspace), _rightsweep!(m, alg, workspace)))


compute!(m::MPOMPOIterativeMultCache, alg::AbstractMPSArith, workspace = scalar_type(m)[]) = iterative_compute!(m, alg, workspace)

function iterative_mult(mpo::MPO, mps::MPO, alg::OneSiteIterativeArith = OneSiteIterativeArith())
    T = promote_type(scalar_type(mpo), scalar_type(mps))
    mpsout = randommpo(T, ophysical_dimensions(mpo), iphysical_dimensions(mps), D=alg.D)
    # canonicalize!(mpsout, normalize=true)
    rightorth!(mpsout, alg=QRFact())
    m = MPOMPOIterativeMultCache(mpo, mps, mpsout, init_hstorage_right(mpsout, mpo, mps))
    kvals = compute!(m, alg)
    return m.ompo, kvals[end]
end


function _leftsweep!(m::MPOMPOIterativeMultCache, alg::OneSiteIterativeArith, workspace = scalar_type(m)[])
    mpoA = m.impo
    mpo = m.mpo
    mpoB = m.ompo
    Cstorage = m.hstorage
    L = length(mpo)
    kvals = Float64[]
    for site in 1:L-1
        (alg.verbosity > 3) && println("Sweeping from left to right at bond: $site.")
        mpsj = reduceH_single_site(mpoA[site], mpo[site], Cstorage[site], Cstorage[site+1])
        push!(kvals, norm(mpsj))
        (alg.verbosity > 1) && println("residual is $(kvals[end])...")
		q, r = tqr!(mpsj, (1,2,4), (3,), workspace)
        mpoB[site] = permute(q, (1,2,4,3))
        Cstorage[site+1] = updateleft(Cstorage[site], mpoB[site], mpo[site], mpoA[site])
    end
    return kvals	
end

function _rightsweep!(m::MPOMPOIterativeMultCache, alg::OneSiteIterativeArith, workspace = scalar_type(m)[])
    mpoA = m.impo
    mpo = m.mpo
    mpoB = m.ompo
    Cstorage = m.hstorage
    L = length(mpo)
    kvals = Float64[]
    r = zeros(scalar_type(mpoB), 0, 0)
    for site in L:-1:2
        (alg.verbosity > 3) && println("Sweeping from right to left at bond: $site.")
        mpsj = reduceH_single_site(mpoA[site], mpo[site], Cstorage[site], Cstorage[site+1])
        push!(kvals, norm(mpsj))
        (alg.verbosity > 1) && println("residual is $(kvals[end])...")
        if isa(alg.fact, QRFact)
        	r, mpoB[site] = tlq!(mpsj, (1,), (2,3,4), workspace)
        elseif isa(alg.fact, SVDFact)
            u, s, v, err = tsvd!(mpsj, (1,), (2,3,4), workspace, trunc=alg.fact.trunc)
            mpoB[site] = v
            r = u * Diagonal(s)
        else
            error("unsupported factorization method $(typeof(alg.fact))")
        end
        Cstorage[site] = updateright(Cstorage[site+1], mpoB[site], mpo[site], mpoA[site])
    end
    # println("norm of r is $(norm(r))")
    mpoB[1] = @tensor tmp[1,2,5,4] := mpoB[1][1,2,3,4] * r[3,5]
    return kvals	
end

function reduceH_single_site(A::AbstractArray{<:Number}, m::AbstractArray{<:Number, 4}, cleft::AbstractArray{<:Number, 3}, cright::AbstractArray{<:Number, 3})
	@tensor tmp[1,7,9,6] := ((cleft[1,2,3] * A[3,4,5,6]) * m[2,7,8,4]) * cright[9,8,5]
    return tmp
end


function init_hstorage_right(B::MPO, mpo::MPO, A::MPO)
    @assert length(B) == length(mpo) == length(A)
    L = length(mpo)
    T = promote_type(scalar_type(B), scalar_type(mpo), scalar_type(A))
    hstorage = Vector{Array{T, 3}}(undef, L+1)
    hstorage[1] = ones(1,1,1)
    hstorage[L+1] = ones(1,1,1)
    for i in L:-1:2
        hstorage[i] = updateright(hstorage[i+1], B[i], mpo[i], A[i])
    end
    return hstorage
end

