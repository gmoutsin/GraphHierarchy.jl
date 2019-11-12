using LinearAlgebra
using LightGraphs
using GraphHierarchy
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

    g = DiGraph(10)
    add_edge!(g,1,2)
    add_edge!(g,2,3)
    add_edge!(g,3,1)
    add_edge!(g,1,4)
    add_edge!(g,3,5)
    add_edge!(g,4,6)
    add_edge!(g,6,5)
    add_edge!(g,10,8)
    add_edge!(g,9,10)
    add_edge!(g,8,4)
    add_edge!(g,7,10)
    add_edge!(g,4,7)
    add_edge!(g,6,9)
    add_edge!(g,5,6)
    HS = graphHierarchicalStructure(g)
    pos = findall(!iszero,HS[3])
    coeffs = graphHierarchicalCoefficients(HS)
    @test norm(HS[1] - [-4.24285714285708, -4.242857142857171, -4.242857142856976,  0.8999999999996752, -0.5285714285718714,  1.1857142857147973,  1.9000000000000274,  4.042857142857218,  2.185714285714233,  3.042857142857145]) < 1.0e-11
    @test norm(HS[2] - [1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]) < 1.0e-11
    @test norm([p[1] for p in pos] - [3,1,2,1,8,3,6,4,5,4,10,6,7,9]) < 1.0e-15
    @test norm([p[2] for p in pos] -[1,2,3,4,4,5,5,6,6,7,8,9,10,10]) < 1.0e-15
    @test norm([HS[3][p] for p in pos] - [0,0,0,5.142857142856755,-3.142857142857543,3.7142857142851042,-1.7142857142866688,0.28571428571512214,1.7142857142866688,1,1,1,1.1428571428571175,0.8571428571429118]) < 1.0e-11
    @test abs(coeffs[1]-3.0/14.0) < 1.0e-11
    @test abs(coeffs[2]-2.0092017518778356) < 1.0e-11
end
