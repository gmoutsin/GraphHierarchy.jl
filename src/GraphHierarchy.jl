module GraphHierarchy

using LightGraphs
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Statistics


function graphHierarchicalStructure(g::SimpleDiGraph{T} where T <: Int)
    A = adjacency_matrix(g)
    d = vec(sum(A, dims=1))
    M = spdiagm(0 => d)-transpose(A)
    dropzeros!(M)
    HLs = lsqr(M,d)
    pos = findall(!iszero,A)
    HDs = sparse([p[1] for p in pos], [p[2] for p in pos], [HLs[p[2]] - HLs[p[1]] for p in pos], length(d), length(d))
    HDsums = sum(HDs, dims=1)
    ICs = [d[i] == 0 ? 1.0 : 1.0 - HDsums[i]/d[i] for i in 1:length(d)]
    return (HLs, ICs, HDs)
end

function graphHierarchicalStructure(g::SimpleGraph{T} where T <: Int)
    A = adjacency_matrix(g)
    d = vec(sum(A, dims=1))
    M = spdiagm(0 => d)-A
    dropzeros!(M)
    HLs = lsqr(M,d)
    pos = findall(!iszero,A)
    HDs = sparse([p[1] for p in pos], [p[2] for p in pos], [HLs[p[2]] - HLs[p[1]] for p in pos], length(d), length(d))
    HDsums = sum(HDs, dims=1)
    ICs = [d[i] == 0 ? 1.0 : 1.0 - HDsums[i]/d[i] for i in 1:length(d)]
    return (HLs, ICs, HDs)
end

function graphHierarchicalStructure(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int)
    d = vec(sum(A, dims=1))
    M = spdiagm(0 => d)-transpose(A)
    dropzeros!(M)
    HLs = lsqr(M,d)
    pos = findall(!iszero,A)
    HDs = sparse([p[1] for p in pos], [p[2] for p in pos], [HLs[p[2]] - HLs[p[1]] for p in pos], length(d), length(d))
    HDsums = sum(HDs, dims=1)
    ICs = [d[i] == 0 ? 1.0 : 1.0 - HDsums[i]/d[i] for i in 1:length(d)]
    return (HLs, ICs, HDs)
end

function graphHierarchicalCoefficients(diffs::SparseMatrixCSC{T1,T2} where T1 <: AbstractFloat where T2 <: Int )
    diffarray = [diffs[p] for p in findall(!iszero, diffs)]
    av = mean(diffarray)
    stdev = std(diffarray, mean=av)
    return (av, stdev)
end

function graphHierarchicalCoefficients(hstruc::Tuple{Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}} where T1 <: AbstractFloat where T2 <: Int)
    diffarray = [hstruc[3][p] for p in findall(!iszero, hstruc[3])]
    av = mean(diffarray)
    stdev = std(diffarray, mean=av)
    return (av, stdev)
end


export graphHierarchicalStructure

export graphHierarchicalCoefficients

end # module
