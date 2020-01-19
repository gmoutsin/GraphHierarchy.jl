module GraphHierarchy

using LightGraphs
using SimpleWeightedGraphs
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using StatsBase
using Statistics

export forward_hierarchical_structure
export backward_hierarchical_structure
export hierarchical_structure
export hierarchical_coefficients

function calculatingStructureAssumingTransposedMatrix(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int)
    d = vec(sum(A, dims=2))
    M = spdiagm(0 => d)-A
    dropzeros!(M)
    HLs = lsqr(M,d)
    pos = findall(!iszero,A)
    HDs = sparse([p[2] for p in pos], [p[1] for p in pos], [HLs[p[1]] - HLs[p[2]] for p in pos], length(d), length(d))
    HDsums = sum(HDs, dims=1)
    ICs = [d[i] == 0 ? 1.0 : 1.0 - sum(HDs[:,i])/d[i] for i in 1:length(d)]
    return (HLs, ICs, HDs )
end

function calculatingStructureAssumingTransposedMatrixBig(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int)
    (x, y, v) = findnz(A)
    vb = -map(BigFloat,v)
    M = sparse(x, y, vb, size(A)[1], size(A)[1])
    d = -vec(sum(M, dims=2))
    for i in 1:length(d)
        if d[i] > 0
            M[i,i] = d[i]
        end
    end
    HLs = lsqr(M,d)
    HDs = sparse(y, x, [ HLs[x[i]] - HLs[y[i]] for i in 1:length(x) ], length(d), length(d) )
    ICs = [d[i] > 0 ? BigFloat(1) - sum(HDs[:,i])/d[i] : BigFloat(1) for i in 1:length(d)]
    return (HLs, ICs, HDs )
end

function calculatingStructureAssumingTransposedWeightedMatrix(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int)
    (x, y, v) = findnz(A)
    d = vec(sum(A, dims=2))
    M = spdiagm(0 => d)-A
    dropzeros!(M)
    HLs = lsqr(M,d)
    HDs = sparse(y, x, [ HLs[x[i]] - HLs[y[i]] for i in 1:length(x) ], length(d), length(d) )
    HDsums = sum(HDs, dims=1)
    ICs = [d[i] > 0 ? 1.0 - A[i,:]'*HDs[:,i]/d[i] : 1.0 for i in 1:length(d)]
    return (HLs, ICs, sparse(HDs) )
end

function calculatingStructureAssumingTransposedWeightedMatrixBig(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int)
    (x, y, v) = findnz(A)
    vb = -map(BigFloat,v)
    M = sparse(x, y, vb, size(A)[1], size(A)[1])
    d = -vec(sum(M, dims=2))
    for i in 1:length(d)
        if d[i] > 0
            M[i,i] = d[i]
        end
    end
    HLs = lsqr(M,d)
    HDs = sparse(y, x, [ HLs[x[i]] - HLs[y[i]] for i in 1:length(x) ], length(d), length(d) )
    HDsums = sum(HDs, dims=1)
    ICs = [d[i] > 0 ? BigFloat(1) - A[i,:]'*HDs[:,i]/d[i] : BigFloat(1) for i in 1:length(d)]
    return (HLs, ICs, HDs )
end

function forward_hierarchical_structure(g::SimpleDiGraph{T} where T <: Int; big::Bool=false)
    if big
        return calculatingStructureAssumingTransposedMatrixBig(sparse(transpose(adjacency_matrix(g))))
    else
        return calculatingStructureAssumingTransposedMatrix(sparse(transpose(adjacency_matrix(g))))
    end
end

function backward_hierarchical_structure(g::SimpleDiGraph{T} where T <: Int; big::Bool=false)
    if big
        return calculatingStructureAssumingTransposedMatrixBig(adjacency_matrix(g))
    else
        return calculatingStructureAssumingTransposedMatrix(adjacency_matrix(g))
    end
end

function hierarchical_structure(g::SimpleDiGraph{T} where T <: Int; big::Bool=false)
    return (forward_hierarchical_structure(g, big = big),backward_hierarchical_structure(g, big = big))
end

function forward_hierarchical_structure(g::SimpleWeightedDiGraph{T1,T2} where T1 <: Int where T2 <: Number; big::Bool=false, weighted::Bool=true)
    if weighted
        if big
            return calculatingStructureAssumingTransposedWeightedMatrixBig(sparse(transpose(LightGraphs.weights(g))))
        else
            return calculatingStructureAssumingTransposedWeightedMatrix(sparse(transpose(LightGraphs.weights(g))))
        end
    else
        if big
            return calculatingStructureAssumingTransposedMatrixBig(sparse(transpose(adjacency_matrix(g))))
        else
            return calculatingStructureAssumingTransposedMatrix(sparse(transpose(adjacency_matrix(g))))
        end
    end
end

function backward_hierarchical_structure(g::SimpleWeightedDiGraph{T1,T2} where T1 <: Int where T2 <: Number; big::Bool=false, weighted::Bool=true)
    if weighted
        if big
            return calculatingStructureAssumingTransposedWeightedMatrixBig(sparse(LightGraphs.weights(g)))
        else
            return calculatingStructureAssumingTransposedWeightedMatrix(sparse(LightGraphs.weights(g)))
        end
    else
        if big
            return calculatingStructureAssumingTransposedMatrixBig(adjacency_matrix(g))
        else
            return calculatingStructureAssumingTransposedMatrix(adjacency_matrix(g))
        end
    end
end

function hierarchical_structure(g::SimpleWeightedDiGraph{T1,T2} where T1 <: Int where T2 <: Number; big::Bool=false, weighted::Bool=true)
    return (forward_hierarchical_structure(g, big = big, weighted = weighted),backward_hierarchical_structure(g, big = big, weighted = weighted))
end

function forward_hierarchical_structure(g::SimpleGraph{T} where T <: Int; big::Bool=false)
    if big
        return calculatingStructureAssumingTransposedMatrixBig(adjacency_matrix(g))
    else
        return calculatingStructureAssumingTransposedMatrix(adjacency_matrix(g))
    end
end

function backward_hierarchical_structure(g::SimpleGraph{T} where T <: Int; big::Bool=false)
    forward_hierarchical_structure(g, big = big)
end

function hierarchical_structure(g::SimpleGraph{T} where T <: Int; big::Bool=false)
    return forward_hierarchical_structure(g, big = big)
end

function forward_hierarchical_structure(g::SimpleWeightedGraph{T1,T2} where T1 <: Int where T2 <: Number; big::Bool=false, weighted::Bool=true)
    if weighted
        if big
            return calculatingStructureAssumingTransposedWeightedMatrixBig(sparse(LightGraphs.weights(g)))
        else
            return calculatingStructureAssumingTransposedWeightedMatrix(sparse(LightGraphs.weights(g)))
        end
    else
        if big
            return calculatingStructureAssumingTransposedMatrixBig(adjacency_matrix(g))
        else
            return calculatingStructureAssumingTransposedMatrix(adjacency_matrix(g))
        end
    end
end

function backward_hierarchical_structure(g::SimpleWeightedGraph{T1,T2} where T1 <: Int where T2 <: Number; big::Bool=false, weighted::Bool=true)
    return forward_hierarchical_structure(g, big = big, weighted = weighted)
end

function hierarchical_structure(g::SimpleWeightedGraph{T1,T2} where T1 <: Int where T2 <: Number; big::Bool=false, weighted::Bool=true)
    return forward_hierarchical_structure(g, big = big, weighted = weighted)
end

function forward_hierarchical_structure(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int; big::Bool=false)
    if big
        return calculatingStructureAssumingTransposedMatrixBig(sparse(transpose(A)))
    else
        return calculatingStructureAssumingTransposedMatrix(sparse(transpose(A)))
    end
end

function backward_hierarchical_structure(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int; big::Bool=false)
    if big
        return calculatingStructureAssumingTransposedMatrixBig(A)
    else
        return calculatingStructureAssumingTransposedMatrix(A)
    end
end

function hierarchical_structure(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int; big::Bool=false)
    return ( forward_hierarchical_structure(A, big=big), backward_hierarchical_structure(A, big=big) )
end

function hierarchical_coefficients(g::Union{SimpleGraph{T},SimpleDiGraph{T}} where T <: Int, diffs::SparseMatrixCSC{T1,T2} where T1 <: AbstractFloat where T2 <: Int)
    diffarray = findnz(diffs)[3]
    av = mean(diffarray)
    return (1 - av, std(diffarray, mean=av, corrected=false))
end

function hierarchical_coefficients(g::Union{SimpleWeightedGraph{T1,T2},SimpleWeightedDiGraph{T1,T2}} where T1 <: Int where T2 <: AbstractFloat, diffs::SparseMatrixCSC{T1,T2} where T1 <: AbstractFloat where T2 <: Int; weighted::Bool = true )
    if weighted
        W0 = sparse(LightGraphs.weights(g))
        (xw,yw,vw) = findnz(W0)
        (x,y,v) = findnz(diffs)

        if sum(map(abs,x - xw)) + sum(map(abs,y - yw)) > 0
            W0 = sparse(transpose(LightGraphs.weights(g)))
            (xw,yw,vw) = findnz(W0)
        end

        if sum(map(abs,x - xw)) + sum(map(abs,y - yw)) > 0
            error("hierarchical differences do not match the weighted adjacency matrix:\n", diffs, "\n", W0)
        end

        w0 = [ W0[x[i],y[i]] for i in 1:length(x)]
        difftype = typeof(v[1])
        if difftype == typeof(w0[1])
            av0 = mean(v, StatsBase.weights(w0))
            return ( 1 - av0 ,std( v, StatsBase.weights(w0), mean=av0, corrected=false))
        else
            w = [ convert( difftype, c) for c in w0 ]
            av = mean(v, StatsBase.weights(w))
            return ( 1 - av ,std( v, StatsBase.weights(w), mean=av, corrected=false))
        end
    else
        v = findnz(diffs)[3]
        av = mean(v)
        return (1 - av, std(v, mean=av))
    end
end

function hierarchical_coefficients(g::Union{SimpleGraph{T},SimpleDiGraph{T}} where T <: Int, hstruc::Tuple{Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}} where T1 <: AbstractFloat where T2 <: Int )
    return hierarchical_coefficients(g, hstruc[3])
end

function hierarchical_coefficients(g::Union{SimpleWeightedGraph{T1,T2},SimpleWeightedDiGraph{T1,T2}} where T1 <: Int  where T2 <: AbstractFloat, hstruc::Tuple{Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}} where T1 <: AbstractFloat where T2 <: Int;  weighted::Bool = true  )
    return hierarchical_coefficients(g, hstruc[3], weighted=weighted)
end

function hierarchical_coefficients(g::Union{SimpleGraph{T},SimpleDiGraph{T}} where T <: Int , hstruc::Tuple{Tuple{Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}},Tuple{Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}}} where T1 <: AbstractFloat where T2 <: Int )
    return (hierarchical_coefficients(g, hstruc[1]),hierarchical_coefficients(g, hstruc[2]))
end

function hierarchical_coefficients(g::Union{SimpleWeightedGraph{T1,T2},SimpleWeightedDiGraph{T1,T2}} where T1 <: Int  where T2 <: AbstractFloat , hstruc::Tuple{Tuple{Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}},Tuple{Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}}} where T1 <: AbstractFloat where T2 <: Int; weighted::Bool = true  )
    return (hierarchical_coefficients(g, hstruc[1], weighted=weighted),hierarchical_coefficients(g, hstruc[2], weighted=weighted))
end

end # module
