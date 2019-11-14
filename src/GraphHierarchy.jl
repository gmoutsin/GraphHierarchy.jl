module GraphHierarchy

using LightGraphs
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Statistics

function calculationAssumingTransposedMatrix(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int)
    d = vec(sum(A, dims=2))
    M = spdiagm(0 => d)-A
    dropzeros!(M)
    HLs = lsqr(M,d)
    pos = findall(!iszero,A)
    HDs = sparse([p[2] for p in pos], [p[1] for p in pos], [HLs[p[1]] - HLs[p[2]] for p in pos], length(d), length(d))
    HDsums = sum(HDs, dims=1)
    ICs = [d[i] == 0 ? 1.0 : 1.0 - HDsums[i]/d[i] for i in 1:length(d)]
    return (HLs, ICs, HDs)
end


function graphHierarchicalStructure(g::SimpleDiGraph{T} where T <: Int)
    return calculationAssumingTransposedMatrix(sparse(transpose(adjacency_matrix(g))))
end

function graphHierarchicalStructure(g::SimpleGraph{T} where T <: Int)
    return calculationAssumingTransposedMatrix(adjacency_matrix(g))
end

function graphHierarchicalStructure(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int)
    return calculationAssumingTransposedMatrix(sparse(transpose(A)))
end

function graphHierarchicalCoefficients(diffs::SparseMatrixCSC{T1,T2} where T1 <: AbstractFloat where T2 <: Int )
    diffarray = findnz(diffs)[3] #[diffs[p] for p in findall(!iszero, diffs)]
    av = mean(diffarray)
    stdev = std(diffarray, mean=av)
    return (1 - av, stdev)
end

function graphHierarchicalCoefficients(hstruc::Tuple{Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}} where T1 <: AbstractFloat where T2 <: Int)
    return graphHierarchicalCoefficients(hstruc[3])
end

export graphHierarchicalStructure

export graphHierarchicalCoefficients

end # module
