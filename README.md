*(README.md is under construction)*

# GraphHierarchy

This package computes the Hierarchical structure of a graph, which was introduced [here](https://arxiv.org/abs/1908.04358).

The module exports two functions `graphHierarchicalStructure` and `graphHierarchicalCoefficients`.

The function `graphHierarchicalStructure` takes a `LightGraphs.jl` graph, a `LightGraphs.jl` digraph or a sparse matrix as an argument and an optional boolean argument If the boolean value is set to `true`, then the result is given as `BigFloat`. The function has the following has 3 signatures:

`function graphHierarchicalStructure(g::SimpleDiGraph{T} where T <: Int, big::Bool=false)`

`function graphHierarchicalStructure(g::SimpleGraph{T} where T <: Int, big::Bool=false)`

`function graphHierarchicalStructure(A::SparseMatrixCSC{T1,T2} where T1 <: Number where T2 <: Int, big::Bool=false)`

It return the `(HL,IC,HD)` where `HL` is the vector of the hierarchical levels of the vertices, `IC` is the vector of the influence centralities of the vertices and `HD` is a sparse matrix with the hierarchical differences of each edge.

The function `graphHierarchicalCoefficients` takes as an argument the tuple `(HL,IC,HD)` or just `HD` and returns `(dc,lc)` where `dc` is the democracy coefficient and `lc` the layering coefficient of the graph.



[![Build Status](https://travis-ci.com/gmoutsin/GraphHierarchy.jl.svg?branch=master)](https://travis-ci.com/gmoutsin/GraphHierarchy.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/gmoutsin/GraphHierarchy.jl?svg=true)](https://ci.appveyor.com/project/gmoutsin/GraphHierarchy-jl)
[![Codecov](https://codecov.io/gh/gmoutsin/GraphHierarchy.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gmoutsin/GraphHierarchy.jl)
[![Coveralls](https://coveralls.io/repos/github/gmoutsin/GraphHierarchy.jl/badge.svg?branch=master)](https://coveralls.io/github/gmoutsin/GraphHierarchy.jl?branch=master)
