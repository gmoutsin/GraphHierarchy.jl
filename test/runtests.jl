using LinearAlgebra
using LightGraphs
using SimpleWeightedGraphs
using GraphHierarchy
using SparseArrays
using Test

# (1,1)-(2,2)

@testset "GraphHierarchy.jl" begin
    # 3-chain
    gw = SimpleWeightedDiGraph(3)
    add_edge!(gw,1,2,2)
    add_edge!(gw,2,3,2)
    # forward: not weighted, not big
    HS = forward_hierarchical_structure(gw, weighted = false)
    @test norm(HS[1] - [-1,0,1]) < 1.0e-12
    @test norm(HS[2] - [1,0,0]) < 1.0e-12
    @test norm(HS[3] - [1,1,0]) < 1.0e-12
    @test norm(HS[4] - [ 0 1 0 ; 0 0 1 ; 0 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1]) < 1.0e-12
    @test abs(HC[2]) < 1.0e-12
    # forward: weighted, not big
    HS = forward_hierarchical_structure(gw)
    @test norm(HS[1] - [-1,0,1]) < 1.0e-12
    @test norm(HS[2] - [1,0,0]) < 1.0e-12
    @test norm(HS[3] - [1,1,0]) < 1.0e-12
    @test norm(HS[4] - [ 0 1 0 ; 0 0 1 ; 0 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1]) < 1.0e-12
    @test abs(HC[2]) < 1.0e-12
    # forward: not weighted, big
    HS = forward_hierarchical_structure(gw, weighted = false, big = true)
    @test norm(HS[1] - [-1,0,1]) < 1.0e-70
    @test norm(HS[2] - [1,0,0]) < 1.0e-70
    @test norm(HS[3] - [1,1,0]) < 1.0e-70
    @test norm(HS[4] - [ 0 1 0 ; 0 0 1 ; 0 0 0 ]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1]) < 1.0e-70
    @test abs(HC[2]) < 1.0e-70
    # forward: weighted, big
    HS = forward_hierarchical_structure(gw, big = true)
    @test norm(HS[1] - [-1,0,1]) < 1.0e-70
    @test norm(HS[2] - [1,0,0]) < 1.0e-70
    @test norm(HS[3] - [1,1,0]) < 1.0e-70
    @test norm(HS[4] - [ 0 1 0 ; 0 0 1 ; 0 0 0 ]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1]) < 1.0e-70
    @test abs(HC[2]) < 1.0e-70
    # backward: not weighted, not big
    HS = backward_hierarchical_structure(gw, weighted = false)
    @test norm(HS[1] - [1,0,-1]) < 1.0e-12
    @test norm(HS[2] - [0,0,1]) < 1.0e-12
    @test norm(HS[3] - [0,1,1]) < 1.0e-12
    @test norm(HS[4] - [ 0 1 0 ; 0 0 1 ; 0 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1]) < 1.0e-12
    @test abs(HC[2]) < 1.0e-12
    # backward: weighted, not big
    HS = backward_hierarchical_structure(gw)
    @test norm(HS[1] - [1,0,-1]) < 1.0e-12
    @test norm(HS[2] - [0,0,1]) < 1.0e-12
    @test norm(HS[3] - [0,1,1]) < 1.0e-12
    @test norm(HS[4] - [ 0 1 0 ; 0 0 1 ; 0 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1]) < 1.0e-12
    @test abs(HC[2]) < 1.0e-12
    # backward: not weighted, big
    HS = backward_hierarchical_structure(gw, weighted = false, big = true)
    @test norm(HS[1] - [1,0,-1]) < 1.0e-70
    @test norm(HS[2] - [0,0,1]) < 1.0e-70
    @test norm(HS[3] - [0,1,1]) < 1.0e-70
    @test norm(HS[4] - [ 0 1 0 ; 0 0 1 ; 0 0 0 ]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1]) < 1.0e-70
    @test abs(HC[2]) < 1.0e-70
    # backward: weighted, big
    HS = backward_hierarchical_structure(gw, big = true)
    @test norm(HS[1] - [1,0,-1]) < 1.0e-70
    @test norm(HS[2] - [0,0,1]) < 1.0e-70
    @test norm(HS[3] - [0,1,1]) < 1.0e-70
    @test norm(HS[4] - [ 0 1 0 ; 0 0 1 ; 0 0 0 ]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1]) < 1.0e-70
    @test abs(HC[2]) < 1.0e-70




    # 3-cycle
    gw = SimpleWeightedDiGraph(3)
    add_edge!(gw,1,2,1)
    add_edge!(gw,2,3,1)
    add_edge!(gw,3,1,2)
    # forward: not weighted, not big
    HS = forward_hierarchical_structure(gw, weighted = false)
    @test norm(HS[1]) < 1.0e-12
    @test norm(HS[2] - [1,1,1]) < 1.0e-12
    @test norm(HS[3]) < 1.0e-12
    @test norm(HS[4]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1] - 1) < 1.0e-12
    @test abs(HC[2]) < 1.0e-12
    # forward: weighted, not big
    HS = forward_hierarchical_structure(gw)
    @test norm(HS[1] - [1//3,0,-1//3]) < 1.0e-12
    @test norm(HS[2] - [1//3, 4//3, 4//3]) < 1.0e-12
    @test norm(HS[3] - [-1//3,-1//3,2//3]) < 1.0e-12
    @test norm(HS[4] - [ 0 -1//3 0 ; 0 0 -1//3 ; 2//3 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1] - 5//6) < 1.0e-12
    @test abs(HC[2] - 1//2) < 1.0e-12
    # forward: not weighted, big
    HS = forward_hierarchical_structure(gw, weighted = false, big = true)
    @test norm(HS[1]) < 1.0e-70
    @test norm(HS[2] - [1,1,1]) < 1.0e-70
    @test norm(HS[3]) < 1.0e-70
    @test norm(HS[4]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1] - 1) < 1.0e-70
    @test abs(HC[2]) < 1.0e-70
    # forward: weighted, big
    HS = forward_hierarchical_structure(gw, big = true)
    @test norm(HS[1] - [1//3,0,-1//3]) < 1.0e-70
    @test norm(HS[2] - [1//3, 4//3, 4//3]) < 1.0e-70
    @test norm(HS[3] - [-1//3,-1//3,2//3]) < 1.0e-70
    @test norm(HS[4] - [ 0 -1//3 0 ; 0 0 -1//3 ; 2//3 0 0 ]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1] - 5//6) < 1.0e-70
    @test abs(HC[2] - 1//2) < 1.0e-70
    # backward: not weighted, not big
    HS = backward_hierarchical_structure(gw, weighted = false)
    @test norm(HS[1]) < 1.0e-12
    @test norm(HS[2] - [1,1,1]) < 1.0e-12
    @test norm(HS[3]) < 1.0e-12
    @test norm(HS[4]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1] - 1) < 1.0e-12
    @test abs(HC[2]) < 1.0e-12
    # # backward: weighted, not big
    HS = backward_hierarchical_structure(gw)
    @test norm(HS[1] - [-1//3,0,1//3]) < 1.0e-12
    @test norm(HS[2] - [4//3,4//3,1//3]) < 1.0e-12
    @test norm(HS[3] - [2//3,-1//3,-1//3]) < 1.0e-12
    @test norm(HS[4] - [ 0 -1//3 0 ; 0 0 -1//3 ; 2//3 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1] - 5//6) < 1.0e-12
    @test abs(HC[2] - 1//2) < 1.0e-12
    # backward: not weighted, big
    HS = backward_hierarchical_structure(gw, weighted = false, big = true)
    @test norm(HS[1]) < 1.0e-70
    @test norm(HS[2] - [1,1,1]) < 1.0e-70
    @test norm(HS[3]) < 1.0e-70
    @test norm(HS[4]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1] - 1) < 1.0e-70
    @test abs(HC[2]) < 1.0e-70
    # backward: weighted, big
    HS = backward_hierarchical_structure(gw, big = true)
    @test norm(HS[1] - [-1//3,0,1//3]) < 1.0e-70
    @test norm(HS[2] - [4//3,4//3,1//3]) < 1.0e-70
    @test norm(HS[3] - [2//3,-1//3,-1//3]) < 1.0e-70
    @test norm(HS[4] - [ 0 -1//3 0 ; 0 0 -1//3 ; 2//3 0 0 ]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1] - 5//6) < 1.0e-70
    @test abs(HC[2] - 1//2) < 1.0e-70




    # pair with stem
    gw = SimpleWeightedDiGraph(3)
    add_edge!(gw,1,2,2)
    add_edge!(gw,2,3,1)
    add_edge!(gw,2,1,1)
    # forward: not weighted, not big
    HS = forward_hierarchical_structure(gw, weighted = false)
    @test norm(HS[1] - [-1//3, -1//3, 2//3]) < 1.0e-12
    @test norm(HS[2] - [1,1,0]) < 1.0e-12
    @test norm(HS[3] - [0,1//2,0]) < 1.0e-12
    @test norm(HS[4] - [ 0 0 0 ; 0 0 1 ; 0 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1] - 2//3) < 1.0e-12
    @test abs(HC[2] - sqrt(2)/3) < 1.0e-12
    # forward: weighted, not big
    HS = forward_hierarchical_structure(gw)
    @test norm(HS[1] - [-11//15,-2//15,13//15]) < 1.0e-12
    @test norm(HS[2] - [8//5,2//5,0]) < 1.0e-12
    @test norm(HS[3] - [3//5,1//5,0]) < 1.0e-12
    @test norm(HS[4] - [ 0 3//5 0 ; -3//5 0 1 ; 0 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1] - 3//5) < 1.0e-12
    @test abs(HC[2] - 3//5) < 1.0e-12
    # forward: not weighted, big
    HS = forward_hierarchical_structure(gw, weighted = false, big = true)
    @test norm(HS[1] - [-1//3, -1//3, 2//3]) < 1.0e-70
    @test norm(HS[2] - [1,1,0]) < 1.0e-70
    @test norm(HS[3] - [0,1//2,0]) < 1.0e-70
    @test norm(HS[4] - [ 0 0 0 ; 0 0 1 ; 0 0 0 ]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1] - 2//3) < 1.0e-70
    @test abs(HC[2] - sqrt(big(2))/3) < 1.0e-70
    # forward: weighted, big
    HS = forward_hierarchical_structure(gw, big = true)
    @test norm(HS[1] - [-11//15,-2//15,13//15]) < 1.0e-70
    @test norm(HS[2] - [8//5,2//5,0]) < 1.0e-70
    @test norm(HS[3] - [3//5,1//5,0]) < 1.0e-70
    @test norm(HS[4] - [ 0 3//5 0 ; -3//5 0 1 ; 0 0 0 ]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1] - 3//5) < 1.0e-70
    @test abs(HC[2] - 3//5) < 1.0e-70
    # backward: not weighted, not big
    HS = backward_hierarchical_structure(gw, weighted = false)
    @test norm(HS[1] - [5//3,2//3,-7//3]) < 1.0e-12
    @test norm(HS[2] - [0,0,1]) < 1.0e-12
    @test norm(HS[3] - [-1,1,3]) < 1.0e-12
    @test norm(HS[4] - [ 0 1 0 ; -1 0 3 ; 0 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1]) < 1.0e-12
    @test abs(HC[2] - 2*sqrt(2/3)) < 1.0e-12
    # backward: weighted, not big
    HS = backward_hierarchical_structure(gw)
    @test norm(HS[1] - [5//3,2//3,-7//3]) < 1.0e-12
    @test norm(HS[2] - [0,0,1]) < 1.0e-12
    @test norm(HS[3] - [-1,1,3]) < 1.0e-12
    @test norm(HS[4] - [ 0 1 0 ; -1 0 3 ; 0 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1]) < 1.0e-12
    @test abs(HC[2] - sqrt(2)) < 1.0e-12
    # backward: not weighted, big
    HS = backward_hierarchical_structure(gw, weighted = false, big = true)
    @test norm(HS[1] - [5//3,2//3,-7//3]) < 1.0e-70
    @test norm(HS[2] - [0,0,1]) < 1.0e-70
    @test norm(HS[3] - [-1,1,3]) < 1.0e-70
    @test norm(HS[4] - [ 0 1 0 ; -1 0 3 ; 0 0 0 ]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1]) < 1.0e-70
    @test abs(HC[2] - 2*sqrt(big(2)/3)) < 1.0e-70
    # backward: weighted, big
    HS = backward_hierarchical_structure(gw, big = true)
    @test norm(HS[1] - [5//3,2//3,-7//3]) < 1.0e-70
    @test norm(HS[2] - [0,0,1]) < 1.0e-70
    @test norm(HS[3] - [-1,1,3]) < 1.0e-70
    @test norm(HS[4] - [ 0 1 0 ; -1 0 3 ; 0 0 0 ]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1]) < 1.0e-70
    @test abs(HC[2] - sqrt(big(2))) < 1.0e-70




    # unbalanced triangle
    gw = SimpleWeightedDiGraph(3)
    add_edge!(gw,1,2,0.1)
    add_edge!(gw,2,3,1)
    add_edge!(gw,3,1,1)
    add_edge!(gw,2,1,1)
    # forward: not weighted, not big
    HS = forward_hierarchical_structure(gw, weighted = false)
    @test norm(HS[1] - [7//18,-5//18,-1//9]) < 1.0e-12
    @test norm(HS[2] - [5//12,5//3,5//6]) < 1.0e-12
    @test norm(HS[3] - [-2//3,5//12,1//2]) < 1.0e-12
    @test norm(HS[4] - [ 0 -2//3 0 ; 2//3 0 1//6 ; 1//2 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1] - 5//6) < 1.0e-12
    @test abs(HC[2] - sqrt(19/2)/6) < 1.0e-12
    # forward: weighted, not big
    HS = forward_hierarchical_structure(gw)
    @test norm(HS[1] - [799//1206,-995//1206,98//603]) < 1.0e-12
    @test norm(HS[2] - [5//804,500//201,5//402]) < 1.0e-12
    @test norm(HS[3] - [-299//201,995//804,1//2]) < 1.0e-12
    @test norm(HS[4] - [ 0 -299//201 0 ; 299//201 0 397//402 ; 1//2 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1] - 550//6231) < 1.0e-12
    @test abs(HC[2] - sqrt(13557755)/6231) < 1.0e-12
    # forward: not weighted, big
    HS = forward_hierarchical_structure(gw, weighted = false, big = true)
    @test norm(HS[1] - [7//18,-5//18,-1//9]) < 1.0e-70
    @test norm(HS[2] - [5//12,5//3,5//6]) < 1.0e-70
    @test norm(HS[3] - [-2//3,5//12,1//2]) < 1.0e-70
    @test norm(HS[4] - [ 0 -2//3 0 ; 2//3 0 1//6 ; 1//2 0 0 ]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1] - 5//6) < 1.0e-70
    @test abs(HC[2] - sqrt(big(19)/2)/6) < 1.0e-70
    # forward: weighted, big  ---  Note that since the weights are given as Float64, accuracy is low even when using BigFloat
    HS = forward_hierarchical_structure(gw, big = true)
    @test norm(HS[1] - [799//1206,-995//1206,98//603]) < 1.0e-16
    @test norm(HS[2] - [5//804,500//201,5//402]) < 1.0e-16
    @test norm(HS[3] - [-299//201,995//804,1//2]) < 1.0e-16
    @test norm(HS[4] - [ 0 -299//201 0 ; 299//201 0 397//402 ; 1//2 0 0 ]) < 1.0e-16
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1] - 550//6231) < 1.0e-16
    @test abs(HC[2] - sqrt(13557755)/6231) < 1.0e-16
    # backward: not weighted, not big
    HS = backward_hierarchical_structure(gw, weighted = false)
    @test norm(HS[1] - [-5//18,7//18,-1//9]) < 1.0e-12
    @test norm(HS[2] - [5//3,5//12,5//6]) < 1.0e-12
    @test norm(HS[3] - [5//12,-2//3,1//2]) < 1.0e-12
    @test norm(HS[4] - [ 0 -2//3 0 ; 2//3 0 1//2 ; 1//6 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1] - 5//6) < 1.0e-12
    @test abs(HC[2] - sqrt(19/2)/6) < 1.0e-12
    # # backward: weighted, not big
    HS = backward_hierarchical_structure(gw)
    @test norm(HS[1] - [-995//1206,799//1206,98//603]) < 1.0e-12
    @test norm(HS[2] - [500//201,5//804,5//402]) < 1.0e-12
    @test norm(HS[3] - [995//804,-299//201,1//2]) < 1.0e-12
    @test norm(HS[4] - [ 0 -299//201 0 ; 299//201 0 1//2 ; 397//402 0 0 ]) < 1.0e-12
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1] - 550//6231) < 1.0e-12
    @test abs(HC[2] - sqrt(13557755)/6231) < 1.0e-12
    # backward: not weighted, big
    HS = backward_hierarchical_structure(gw, weighted = false, big = true)
    @test norm(HS[1] - [-5//18,7//18,-1//9]) < 1.0e-70
    @test norm(HS[2] - [5//3,5//12,5//6]) < 1.0e-70
    @test norm(HS[3] - [5//12,-2//3,1//2]) < 1.0e-70
    @test norm(HS[4] - [ 0 -2//3 0 ; 2//3 0 1//2 ; 1//6 0 0 ]) < 1.0e-70
    HC = hierarchical_coefficients(gw, HS, weighted = false)
    @test abs(HC[1] - 5//6) < 1.0e-70
    @test abs(HC[2] - sqrt(big(19)/2)/6) < 1.0e-70
    # backward: weighted, big
    HS = backward_hierarchical_structure(gw, big = true)
    @test norm(HS[1] - [-995//1206,799//1206,98//603]) < 1.0e-16
    @test norm(HS[2] - [500//201,5//804,5//402]) < 1.0e-16
    @test norm(HS[3] - [995//804,-299//201,1//2]) < 1.0e-16
    @test norm(HS[4] - [ 0 -299//201 0 ; 299//201 0 1//2 ; 397//402 0 0  ]) < 1.0e-16
    HC = hierarchical_coefficients(gw, HS)
    @test abs(HC[1] - 550//6231) < 1.0e-16
    @test abs(HC[2] - sqrt(big(13557755))/6231) < 1.0e-16
end
