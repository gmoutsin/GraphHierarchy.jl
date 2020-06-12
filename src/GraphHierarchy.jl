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
export forward_hierarchical_coefficients
export backward_hierarchical_coefficients
export hierarchical_coefficients



function weightedMeanStd(v::Array{T,1} where T <: AbstractFloat, w::Array{T,1} where T <: AbstractFloat)
    difftype = typeof(v[1])
    if difftype == typeof(w[1])
        av0 = mean(v, StatsBase.weights(w))
        return ( 1 - av0 ,std( v, StatsBase.weights(w), mean=av0, corrected=false))
    else
        ww = [ convert( difftype, c) for c in w ]
        av = mean(v, StatsBase.weights(ww))
        return ( 1 - av ,std( v, StatsBase.weights(ww), mean=av, corrected=false))
    end
end


function calculatingStructureAssumingTransposedMatrix(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int)
    d = vec(sum(A, dims=2))
    dout = vec(sum(A, dims=1))
    M = spdiagm(0 => d)-A
    dropzeros!(M)
    HLs = lsqr(M,d)
    pos = findall(!iszero,A)
    HDs = sparse([p[2] for p in pos], [p[1] for p in pos], [HLs[p[1]] - HLs[p[2]] for p in pos], length(d), length(d))
    HDsums = sum(HDs, dims=1)
    ICs = [d[i] > 0 ? 1.0 - sum(HDs[:,i])/d[i] : 1.0 for i in 1:length(d)]
    RCs = [dout[i] > 0 ? sum(HDs[i,:])/dout[i] : 0.0 for i in 1:length(dout)]
    return (HLs, ICs, RCs, HDs )
end


function calculatingStructureAssumingTransposedMatrixBig(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int)
    (x, y, v) = findnz(A)
    vb = -map(BigFloat,v)
    M = sparse(x, y, vb, size(A)[1], size(A)[1])
    d = -vec(sum(M, dims=2))
    dout = -vec(sum(M, dims=1))
    for i in 1:length(d)
        if d[i] > 0
            M[i,i] = d[i]
        end
    end
    HLs = lsqr(M,d)
    HDs = sparse(y, x, [ HLs[x[i]] - HLs[y[i]] for i in 1:length(x) ], length(d), length(d) )
    ICs = [d[i] > 0 ? BigFloat(1) - sum(HDs[:,i])/d[i] : BigFloat(1) for i in 1:length(d)]
    RCs = [dout[i] > 0 ? sum(HDs[i,:])/dout[i] : BigFloat(0)  for i in 1:length(dout)]
    return (HLs, ICs, RCs, HDs )
end


function calculatingStructureAssumingTransposedWeightedMatrix(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int)
    (x, y, v) = findnz(A)
    d = vec(sum(A, dims=2))
    dout = vec(sum(A, dims=1))
    M = spdiagm(0 => d)-A
    dropzeros!(M)
    HLs = lsqr(M,d)
    HDs = sparse(y, x, [ HLs[x[i]] - HLs[y[i]] for i in 1:length(x) ], length(d), length(d) )
    HDsums = sum(HDs, dims=1)
    ICs = [d[i] > 0 ? 1.0 - A[i,:]'*HDs[:,i]/d[i] : 1.0 for i in 1:length(d)]
    RCs = [dout[i] > 0 ? A[:,i]'*HDs[i,:]/dout[i] : 0.0 for i in 1:length(dout)]
    return (HLs, ICs, RCs, HDs )
end


function calculatingStructureAssumingTransposedWeightedMatrixBig(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int)
    (x, y, v) = findnz(A)
    vb = -map(BigFloat,v)
    M = sparse(x, y, vb, size(A)[1], size(A)[1])
    d = -vec(sum(M, dims=2))
    dout = -vec(sum(M, dims=1))
    for i in 1:length(d)
        if d[i] > 0
            M[i,i] = d[i]
        end
    end
    HLs = lsqr(M,d)
    HDs = sparse(y, x, [ HLs[x[i]] - HLs[y[i]] for i in 1:length(x) ], length(d), length(d) )
    HDsums = sum(HDs, dims=1)
    ICs = [d[i] > 0 ? BigFloat(1) - A[i,:]'*HDs[:,i]/d[i] : BigFloat(1) for i in 1:length(d)]
    RCs = [dout[i] > 0 ? A[:,i]'*HDs[i,:]/dout[i] : BigFloat(0) for i in 1:length(dout)]
    return (HLs, ICs, RCs, HDs )
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
        (xb, yb, wb, zb) = calculatingStructureAssumingTransposedMatrixBig(adjacency_matrix(g))
        return (xb, yb, wb, sparse(transpose(zb)))
    else
        (x, y, w, z) = calculatingStructureAssumingTransposedMatrix(adjacency_matrix(g))
        return (x, y, w, sparse(transpose(z)))
    end
end


function hierarchical_structure(g::SimpleDiGraph{T} where T <: Int; big::Bool=false)
    return (forward_hierarchical_structure(g, big = big),backward_hierarchical_structure(g, big = big))
end


function forward_hierarchical_structure(g::SimpleWeightedDiGraph{T1,T2} where T1 <: Int where T2 <: Number; big::Bool=false, weighted::Bool=true)
    if weighted
        if big
            return calculatingStructureAssumingTransposedWeightedMatrixBig(transpose(LightGraphs.weights(g)))
        else
            return calculatingStructureAssumingTransposedWeightedMatrix(transpose(LightGraphs.weights(g)))
        end
    else
        if big
            return calculatingStructureAssumingTransposedMatrixBig(sparse(transpose((!iszero).(adjacency_matrix(g)))))
        else
            return calculatingStructureAssumingTransposedMatrix(sparse(transpose((!iszero).(adjacency_matrix(g)))))
        end
    end
end


function backward_hierarchical_structure(g::SimpleWeightedDiGraph{T1,T2} where T1 <: Int where T2 <: Number; big::Bool=false, weighted::Bool=true)
    if weighted
        if big
            (xb, yb, wb, zb) = calculatingStructureAssumingTransposedWeightedMatrixBig(sparse(LightGraphs.weights(g)))
            return (xb, yb, wb, sparse(transpose(zb)))
        else
            (x, y, w, z) = calculatingStructureAssumingTransposedWeightedMatrix(sparse(LightGraphs.weights(g)))
            return (x, y, w, sparse(transpose(z)))
        end
    else
        if big
            (xb, yb, wb, zb) = calculatingStructureAssumingTransposedMatrixBig(sparse((!iszero).(adjacency_matrix(g))))
            return (xb, yb, wb, sparse(transpose(zb)))
        else
            (x, y, w, z) = calculatingStructureAssumingTransposedMatrix(sparse((!iszero).(adjacency_matrix(g))))
            return (x, y, w, sparse(transpose(z)))
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


function hierarchical_coefficients(g::Union{SimpleGraph{T},SimpleDiGraph{T}} where T <: Int, hstruc::Tuple{Array{T1,1},Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}} where T1 <: AbstractFloat where T2 <: Int )
    return hierarchical_coefficients(g, hstruc[4])
end

function hierarchical_coefficients(g::SimpleDiGraph{T} where T <: Int , hstruc::Tuple{Tuple{Array{T1,1},Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}},Tuple{Array{T1,1},Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}}} where T1 <: AbstractFloat where T2 <: Int )
    return (hierarchical_coefficients(g, hstruc[1]),hierarchical_coefficients(g, hstruc[2]))
end


function hierarchical_coefficients(g::Union{SimpleWeightedGraph{T1,T2},SimpleWeightedDiGraph{T1,T2}} where T1 <: Int where T2 <: AbstractFloat, diffs::SparseMatrixCSC{T1,T2} where T1 <: AbstractFloat where T2 <: Int; weighted::Bool = true )
    if weighted
        W0 = sparse(LightGraphs.weights(g))
        (xw,yw,vw) = findnz(W0)
        (x,y,v) = findnz(diffs)

        if sum(map(abs,x - xw)) + sum(map(abs,y - yw)) > 0
            error("Hierarchical differences do not match the weighted adjacency matrix:\nDifferences:", diffs, "\nWeighted Adjacency Matrix:", W0)
        end

        return weightedMeanStd(v,vw)
    else
        v = findnz(diffs)[3]
        av = mean(v)
        return (1 - av, std(v, mean=av, corrected=false))
    end
end

function hierarchical_coefficients(g::Union{SimpleWeightedGraph{T1,T2},SimpleWeightedDiGraph{T1,T2}} where T1 <: Int  where T2 <: AbstractFloat, hstruc::Tuple{Array{T1,1},Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}} where T1 <: AbstractFloat where T2 <: Int;  weighted::Bool = true  )
    return hierarchical_coefficients(g, hstruc[4], weighted=weighted)
end




function hierarchical_coefficients(g::SimpleWeightedDiGraph{T1,T2} where T1 <: Int  where T2 <: AbstractFloat , hstruc::Tuple{Tuple{Array{T1,1},Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}},Tuple{Array{T1,1},Array{T1,1},Array{T1,1},SparseMatrixCSC{T1,T2}}} where T1 <: AbstractFloat where T2 <: Int; weighted::Bool = true  )
    return (hierarchical_coefficients(g, hstruc[1], weighted=weighted), hierarchical_coefficients(g, hstruc[2], weighted=weighted))
end


end # module
