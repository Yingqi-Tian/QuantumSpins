

function prodmpo(physpaces::Vector{Int}, m::QTerm) 
	is_constant(m) || throw(ArgumentError("only constant term allowed."))
	return prodmpo(scalar_type(m), physpaces, positions(m), op(m)) * value(coeff(m))
end

function MPO(h::QuantumOperator; alg::MPOCompression=Deparallelise())
	physpaces = physical_dimensions(h)
	local mpo
	compress_threshold = 20
	for m in qterms(h)
		if @isdefined mpo
			mpo += prodmpo(physpaces, m)
		else
			mpo = prodmpo(physpaces, m)
		end
		if bond_dimension(mpo) >= compress_threshold
			mpo = compress!(mpo, alg=alg)
			compress_threshold += 5
		end
	end
	mpo = compress!(mpo, alg=alg)
	return mpo
end

MPO(h::SuperOperatorBase; kwargs...) = MPO(h.data; kwargs...)
