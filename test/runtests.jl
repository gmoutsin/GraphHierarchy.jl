using GraphHierarchy
using LinearAlgebra
using Test

@testset "GraphHierarchy.jl" begin
    g = DiGraph(6)
    add_edge!(g,1,2)
    add_edge!(g,2,3)
    add_edge!(g,3,4)
    add_edge!(g,4,5)
    add_edge!(g,5,6)
    HS = graphHierarchicalStructure(g)
    pos = findall(!iszero,HS[3])
    coeffs = graphHierarchicalCoefficients(HS)
    @test norm(HS[1] - [-2.5,-1.5,-0.5,0.5,1.5,2.5]) < 1.0e-15
    @test norm(HS[2] - [1,0,0,0,0,0]) < 1.0e-15
    @test norm([p[1] for p in pos] - [1,2,3,4,5]) < 1.0e-15
    @test norm([p[2] for p in pos] - [2,3,4,5,6]) < 1.0e-15
    @test norm([HS[3][p] for p in pos] - [1.0,1.0,1.0,1.0,1.0]) < 1.0e-15
    @test abs(coeffs[1]-0.0) < 1.0e-15
    @test abs(coeffs[2]-0.0) < 1.0e-15

    g = DiGraph(11)
    add_edge!(g,1,3)
    add_edge!(g,1,4)
    add_edge!(g,2,3)
    add_edge!(g,2,4)
    add_edge!(g,3,5)
    add_edge!(g,3,6)
    add_edge!(g,4,5)
    add_edge!(g,4,6)
    add_edge!(g,5,7)
    add_edge!(g,5,8)
    add_edge!(g,6,7)
    add_edge!(g,6,8)
    add_edge!(g,7,9)
    add_edge!(g,7,10)
    add_edge!(g,8,9)
    add_edge!(g,8,10)
    add_edge!(g,9,11)
    add_edge!(g,10,11)
    HS = graphHierarchicalStructure(g)
    pos = findall(!iszero,HS[3])
    coeffs = graphHierarchicalCoefficients(HS)
    @test norm(HS[1] - [ceil(i/2) - 36.0/11.0 for i in 1:11]) < 1.0e-15
    @test norm(HS[2] - [1,1,0,0,0,0,0,0,0,0,0]) < 1.0e-15
    @test norm([p[1] for p in pos] - [1,2,1,2,3,4,3,4,5,6,5,6,7,8,7,8,9,10]) < 1.0e-15
    @test norm([p[2] for p in pos] - [3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11]) < 1.0e-15
    @test norm([HS[3][p] for p in pos] - [1.0 for p in pos]) < 1.0e-15
    @test abs(coeffs[1]-0.0) < 1.0e-15
    @test abs(coeffs[2]-0.0) < 1.0e-15
end
