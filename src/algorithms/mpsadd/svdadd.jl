

@with_kw struct SVDAdd <: AbstractMPSAdd
	D::Int = DefaultTruncation.D
	ϵ::Float64 = DefaultTruncation.ϵ
	verbosity::Int = Defaults.verbosity
end

get_trunc(x::SVDAdd) = MPSTruncation(D=x.D, ϵ=x.ϵ)

function svd_add(psiA::MPS, psiB::MPS, alg::SVDAdd = SVDAdd()) 
    (length(psiA) == length(psiB)) || throw(DimensionMismatch())
    (isempty(psiA)) && error("input mps is empty.")
    if length(psiA) == 1
        return MPS([psiA[1] + psiB[1]])
    end
    trunc = get_trunc(alg)
    T = promote_type(scalar_type(psiA), scalar_type(psiB))
    L = length(psiA)
    r = MPS{T}(L)
    # r[1] = cat(psiA[1], psiB[1], dims=3)
    workspace = T[]
    tmp = cat(psiA[1], psiB[1], dims=3)
    u, s, v, err = tsvd!(tmp, (1, 2), (3,), workspace, trunc=trunc)
    r[1] = u
    v = Diagonal(s) * v
    for i in 2:L-1
    	rj = cat(psiA[i], psiB[i], dims=(1,3))
    	@tensor tmp[1,3,4] := v[1,2] * rj[2,3,4]
    	u, s, v, err = tsvd!(tmp, (1,2), (3,), workspace, trunc=trunc)
    	v = Diagonal(s) * v
    	r[i] = u
    end
    rj = cat(psiA[L], psiB[L], dims=1)
    @tensor tmp[1,3,4] := v[1,2] * rj[2,3,4]
    r[L] = tmp
    rightorth!(r, workspace, trunc=trunc)
    return r, err
end

function svd_add(psis::Vector{<:MPS}, alg::SVDAdd = SVDAdd())
	isempty(psis) && error("no states.")
	(length(psis)==1) && return psis[1]
	r, err = svd_add(psis[1], psis[2], alg)
	for i in 3:length(psis)
		r, err = svd_add(r, psis[i], alg)
	end
	return r, err
end
