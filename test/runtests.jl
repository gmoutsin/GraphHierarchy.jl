using LinearAlgebra
using LightGraphs
using SimpleWeightedGraphs
using GraphHierarchy
using SparseArrays
using Test

# (1,1)-(2,2)

@testset "GraphHierarchy.jl" begin
    gw = SimpleWeightedDiGraph(6)
    add_edge!(gw,1,2,2)
    add_edge!(gw,2,3,2)
    add_edge!(gw,3,4,2)
    add_edge!(gw,4,5,2)
    add_edge!(gw,5,6,2)
    g = SimpleDiGraph(adjacency_matrix(gw))
    HS = forward_hierarchical_structure(g)
    spdata = findnz(HS[3])
    coeffs = hierarchical_coefficients(g,HS)
    @test norm(HS[1] - [-2.5,-1.5,-0.5,0.5,1.5,2.5]) < 1.0e-14
    @test norm(HS[2] - [1,0,0,0,0,0]) < 1.0e-14
    @test norm(spdata[1] - [1,2,3,4,5]) < 1.0e-14
    @test norm(spdata[2] - [2,3,4,5,6]) < 1.0e-14
    @test norm(spdata[3]- [1.0,1.0,1.0,1.0,1.0]) < 1.0e-14
    @test abs(coeffs[1]) < 1.0e-14
    @test abs(coeffs[2]) < 1.0e-14
    HSA = forward_hierarchical_structure(adjacency_matrix(g))
    spdataA = findnz(HSA[3])
    coeffsA = hierarchical_coefficients(g,HSA)
    @test norm(HSA[1] - HS[1]) < 1.0e-14
    @test norm(HSA[2] - HS[2]) < 1.0e-14
    @test norm(spdataA[1] - spdata[1]) < 1.0e-14
    @test norm(spdataA[2] - spdata[2]) < 1.0e-14
    @test norm(spdataA[3] - spdata[3]) < 1.0e-14
    @test abs(coeffsA[1] - coeffs[1]) < 1.0e-14
    @test abs(coeffsA[2] - coeffs[2]) < 1.0e-14
    gr = SimpleDiGraph(transpose(adjacency_matrix(g)))
    HSf = hierarchical_structure(g)
    HSr = hierarchical_structure(gr)
    spdata_f1 = findnz(HSf[1][3])
    spdata_f2 = findnz(HSf[2][3])
    coeffs_f = hierarchical_coefficients(g,HSf)
    spdata_r1 = findnz(HSr[1][3])
    spdata_r2 = findnz(HSr[2][3])
    coeffs_r = hierarchical_coefficients(gr,HSr)
    @test norm(HSf[1][1] - HSr[2][1]) < 1.0e-14
    @test norm(HSf[2][1] - HSr[1][1]) < 1.0e-14
    @test norm(HSf[1][2] - HSr[2][2]) < 1.0e-14
    @test norm(HSf[2][2] - HSr[1][2]) < 1.0e-14
    @test norm(spdata_f1[1] - spdata_r2[1]) < 1.0e-14
    @test norm(spdata_f1[2] - spdata_r2[2]) < 1.0e-14
    @test norm(spdata_f1[3] - spdata_r2[3]) < 1.0e-14
    @test norm(spdata_f2[1] - spdata_r1[1]) < 1.0e-14
    @test norm(spdata_f2[2] - spdata_r1[2]) < 1.0e-14
    @test norm(spdata_f2[3] - spdata_r1[3]) < 1.0e-14
    @test abs(coeffs_f[1][1] - coeffs_r[2][1]) < 1.0e-14
    @test abs(coeffs_f[1][2] - coeffs_r[2][2]) < 1.0e-14
    @test abs(coeffs_f[2][1] - coeffs_r[1][1]) < 1.0e-14
    @test abs(coeffs_f[2][2] - coeffs_r[1][2]) < 1.0e-14
    gwr = SimpleWeightedDiGraph(transpose(weights(gw)))
    HSwf = hierarchical_structure(gw)
    HSwr = hierarchical_structure(gwr)
    spdata_wf1 = findnz(HSwf[1][3])
    spdata_wf2 = findnz(HSwf[2][3])
    coeffs_wf = hierarchical_coefficients(gw,HSwf)
    spdata_wr1 = findnz(HSwr[1][3])
    spdata_wr2 = findnz(HSwr[2][3])
    coeffs_wr = hierarchical_coefficients(gwr,HSwr)
    @test norm(HSwf[1][1] - HSwr[2][1]) < 1.0e-14
    @test norm(HSwf[2][1] - HSwr[1][1]) < 1.0e-14
    @test norm(HSwf[1][2] - HSwr[2][2]) < 1.0e-14
    @test norm(HSwf[2][2] - HSwr[1][2]) < 1.0e-14
    @test norm(spdata_wf1[1] - spdata_wr2[1]) < 1.0e-14
    @test norm(spdata_wf1[2] - spdata_wr2[2]) < 1.0e-14
    @test norm(spdata_wf1[3] - spdata_wr2[3]) < 1.0e-14
    @test norm(spdata_wf2[1] - spdata_wr1[1]) < 1.0e-14
    @test norm(spdata_wf2[2] - spdata_wr1[2]) < 1.0e-14
    @test norm(spdata_wf2[3] - spdata_wr1[3]) < 1.0e-14
    @test abs(coeffs_wf[1][1] - coeffs_wr[2][1]) < 1.0e-14
    @test abs(coeffs_wf[1][2] - coeffs_wr[2][2]) < 1.0e-14
    @test abs(coeffs_wf[2][1] - coeffs_wr[1][1]) < 1.0e-14
    @test abs(coeffs_wf[2][2] - coeffs_wr[1][2]) < 1.0e-14
    @test norm(HSf[1][1] - HSwr[2][1]) < 1.0e-14
    @test norm(HSf[2][1] - HSwr[1][1]) < 1.0e-14
    @test norm(HSf[1][2] - HSwr[2][2]) < 1.0e-14
    @test norm(HSf[2][2] - HSwr[1][2]) < 1.0e-14
    @test norm(spdata_f1[1] - spdata_wr2[1]) < 1.0e-14
    @test norm(spdata_f1[2] - spdata_wr2[2]) < 1.0e-14
    @test norm(spdata_f1[3] - spdata_wr2[3]) < 1.0e-14
    @test norm(spdata_f2[1] - spdata_wr1[1]) < 1.0e-14
    @test norm(spdata_f2[2] - spdata_wr1[2]) < 1.0e-14
    @test norm(spdata_f2[3] - spdata_wr1[3]) < 1.0e-14
    @test abs(coeffs_f[1][1] - coeffs_wr[2][1]) < 1.0e-14
    @test abs(coeffs_f[1][2] - coeffs_wr[2][2]) < 1.0e-14
    @test abs(coeffs_f[2][1] - coeffs_wr[1][1]) < 1.0e-14
    @test abs(coeffs_f[2][2] - coeffs_wr[1][2]) < 1.0e-14

    B_HS = forward_hierarchical_structure(g, big=true)
    B_spdata = findnz(B_HS[3])
    B_coeffs = hierarchical_coefficients(g,B_HS)
    @test norm(B_HS[1] - [-big(5)/2,-big(3)/2,-big(1)/2,big(1)/2,big(3)/2,big(5)/2]) < 1.0e-70
    @test norm(B_HS[2] - [BigFloat(1),BigFloat(0),BigFloat(0),BigFloat(0),BigFloat(0),BigFloat(0)]) < 1.0e-70
    @test norm(B_spdata[3]- [BigFloat(1),BigFloat(1),BigFloat(1),BigFloat(1),BigFloat(1)]) < 1.0e-70
    @test norm(B_HS[1] - HS[1]) < 1.0e-14
    @test norm(B_HS[2] - HS[2]) < 1.0e-14
    @test norm(B_spdata[1] - spdata[1]) < 1.0e-14
    @test norm(B_spdata[2] - spdata[2]) < 1.0e-14
    @test norm(B_spdata[3] - spdata[3]) < 1.0e-14
    @test abs(B_coeffs[1]) < 1.0e-70
    @test abs(B_coeffs[2]) < 1.0e-70
    B_HSA = forward_hierarchical_structure(adjacency_matrix(g), big=true)
    B_spdataA = findnz(B_HSA[3])
    B_coeffsA = hierarchical_coefficients(g,B_HSA)
    @test norm(B_HSA[1] - B_HS[1]) < 1.0e-70
    @test norm(B_HSA[2] - B_HS[2]) < 1.0e-70
    @test norm(B_spdataA[1] - B_spdata[1]) < 1.0e-70
    @test norm(B_spdataA[2] - B_spdata[2]) < 1.0e-70
    @test norm(B_spdataA[3] - B_spdata[3]) < 1.0e-70
    @test abs(B_coeffsA[1] - B_coeffs[1]) < 1.0e-70
    @test abs(B_coeffsA[2] - B_coeffs[2]) < 1.0e-70
    B_HSf = hierarchical_structure(g, big=true)
    B_HSr = hierarchical_structure(gr, big=true)
    B_spdata_f1 = findnz(B_HSf[1][3])
    B_spdata_f2 = findnz(B_HSf[2][3])
    B_coeffs_f = hierarchical_coefficients(g,B_HSf)
    B_spdata_r1 = findnz(B_HSr[1][3])
    B_spdata_r2 = findnz(B_HSr[2][3])
    B_coeffs_r = hierarchical_coefficients(gr,B_HSr)
    @test norm(B_HSf[1][1] - B_HSr[2][1]) < 1.0e-70
    @test norm(B_HSf[2][1] - B_HSr[1][1]) < 1.0e-70
    @test norm(B_HSf[1][2] - B_HSr[2][2]) < 1.0e-70
    @test norm(B_HSf[2][2] - B_HSr[1][2]) < 1.0e-70
    @test norm(B_spdata_f1[1] - B_spdata_r2[1]) < 1.0e-70
    @test norm(B_spdata_f1[2] - B_spdata_r2[2]) < 1.0e-70
    @test norm(B_spdata_f1[3] - B_spdata_r2[3]) < 1.0e-70
    @test norm(B_spdata_f2[1] - B_spdata_r1[1]) < 1.0e-70
    @test norm(B_spdata_f2[2] - B_spdata_r1[2]) < 1.0e-70
    @test norm(B_spdata_f2[3] - B_spdata_r1[3]) < 1.0e-70
    @test abs(B_coeffs_f[1][1] - B_coeffs_r[2][1]) < 1.0e-70
    @test abs(B_coeffs_f[1][2] - B_coeffs_r[2][2]) < 1.0e-70
    @test abs(B_coeffs_f[2][1] - B_coeffs_r[1][1]) < 1.0e-70
    @test abs(B_coeffs_f[2][2] - B_coeffs_r[1][2]) < 1.0e-70
    B_HSwf = hierarchical_structure(gw, big=true)
    B_HSwr = hierarchical_structure(gwr, big=true)
    B_spdata_wf1 = findnz(B_HSwf[1][3])
    B_spdata_wf2 = findnz(B_HSwf[2][3])
    B_coeffs_wf = hierarchical_coefficients(gw,B_HSwf)
    B_spdata_wr1 = findnz(B_HSwr[1][3])
    B_spdata_wr2 = findnz(B_HSwr[2][3])
    B_coeffs_wr = hierarchical_coefficients(gwr,B_HSwr)
    @test norm(B_HSwf[1][1] - B_HSwr[2][1]) < 1.0e-70
    @test norm(B_HSwf[2][1] - B_HSwr[1][1]) < 1.0e-70
    @test norm(B_HSwf[1][2] - B_HSwr[2][2]) < 1.0e-70
    @test norm(B_HSwf[2][2] - B_HSwr[1][2]) < 1.0e-70
    @test norm(B_spdata_wf1[1] - B_spdata_wr2[1]) < 1.0e-70
    @test norm(B_spdata_wf1[2] - B_spdata_wr2[2]) < 1.0e-70
    @test norm(B_spdata_wf1[3] - B_spdata_wr2[3]) < 1.0e-70
    @test norm(B_spdata_wf2[1] - B_spdata_wr1[1]) < 1.0e-70
    @test norm(B_spdata_wf2[2] - B_spdata_wr1[2]) < 1.0e-70
    @test norm(B_spdata_wf2[3] - B_spdata_wr1[3]) < 1.0e-70
    @test abs(B_coeffs_wf[1][1] - B_coeffs_wr[2][1]) < 1.0e-70
    @test abs(B_coeffs_wf[1][2] - B_coeffs_wr[2][2]) < 1.0e-70
    @test abs(B_coeffs_wf[2][1] - B_coeffs_wr[1][1]) < 1.0e-70
    @test abs(B_coeffs_wf[2][2] - B_coeffs_wr[1][2]) < 1.0e-70
    @test norm(B_HSf[1][1] - B_HSwr[2][1]) < 1.0e-70
    @test norm(B_HSf[2][1] - B_HSwr[1][1]) < 1.0e-70
    @test norm(B_HSf[1][2] - B_HSwr[2][2]) < 1.0e-70
    @test norm(B_HSf[2][2] - B_HSwr[1][2]) < 1.0e-70
    @test norm(B_spdata_f1[1] - B_spdata_wr2[1]) < 1.0e-70
    @test norm(B_spdata_f1[2] - B_spdata_wr2[2]) < 1.0e-70
    @test norm(B_spdata_f1[3] - B_spdata_wr2[3]) < 1.0e-70
    @test norm(B_spdata_f2[1] - B_spdata_wr1[1]) < 1.0e-70
    @test norm(B_spdata_f2[2] - B_spdata_wr1[2]) < 1.0e-70
    @test norm(B_spdata_f2[3] - B_spdata_wr1[3]) < 1.0e-70
    @test abs(B_coeffs_f[1][1] - B_coeffs_wr[2][1]) < 1.0e-70
    @test abs(B_coeffs_f[1][2] - B_coeffs_wr[2][2]) < 1.0e-70
    @test abs(B_coeffs_f[2][1] - B_coeffs_wr[1][1]) < 1.0e-70
    @test abs(B_coeffs_f[2][2] - B_coeffs_wr[1][2]) < 1.0e-70


    gw = SimpleWeightedDiGraph(11)
    add_edge!(gw,1,3,2)
    add_edge!(gw,1,4,2)
    add_edge!(gw,2,3,2)
    add_edge!(gw,2,4,2)
    add_edge!(gw,3,5,2)
    add_edge!(gw,3,6,2)
    add_edge!(gw,4,5,2)
    add_edge!(gw,4,6,2)
    add_edge!(gw,5,7,2)
    add_edge!(gw,5,8,2)
    add_edge!(gw,6,7,2)
    add_edge!(gw,6,8,2)
    add_edge!(gw,7,9,2)
    add_edge!(gw,7,10,2)
    add_edge!(gw,8,9,2)
    add_edge!(gw,8,10,2)
    add_edge!(gw,9,11,2)
    add_edge!(gw,10,11,2)
    g = SimpleDiGraph(adjacency_matrix(gw))
    HS = forward_hierarchical_structure(g)
    spdata = findnz(HS[3])
    coeffs = hierarchical_coefficients(g,HS)
    @test norm(HS[1] - [ceil(i/2) - 36.0/11.0 for i in 1:11]) < 1.0e-14
    @test norm(HS[2] - [1,1,0,0,0,0,0,0,0,0,0]) < 1.0e-14
    @test norm(spdata[1] - [1,2,1,2,3,4,3,4,5,6,5,6,7,8,7,8,9,10]) < 1.0e-14
    @test norm(spdata[2] - [3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11]) < 1.0e-14
    @test norm(spdata[3] - [1.0 for i in 1:length(spdata[3])]) < 1.0e-14
    @test abs(coeffs[1]-0.0) < 1.0e-14
    @test abs(coeffs[2]-0.0) < 1.0e-14
    HSA = forward_hierarchical_structure(adjacency_matrix(g))
    spdataA = findnz(HSA[3])
    coeffsA = hierarchical_coefficients(g,HSA)
    @test norm(HSA[1] - HS[1]) < 1.0e-14
    @test norm(HSA[2] - HS[2]) < 1.0e-14
    @test norm(spdataA[1] - spdata[1]) < 1.0e-14
    @test norm(spdataA[2] - spdata[2]) < 1.0e-14
    @test norm(spdataA[3] - spdata[3]) < 1.0e-14
    @test abs(coeffsA[1] - coeffs[1]) < 1.0e-14
    @test abs(coeffsA[2] - coeffs[2]) < 1.0e-14
    gr = SimpleDiGraph(transpose(adjacency_matrix(g)))
    HSf = hierarchical_structure(g)
    HSr = hierarchical_structure(gr)
    spdata_f1 = findnz(HSf[1][3])
    spdata_f2 = findnz(HSf[2][3])
    coeffs_f = hierarchical_coefficients(g,HSf)
    spdata_r1 = findnz(HSr[1][3])
    spdata_r2 = findnz(HSr[2][3])
    coeffs_r = hierarchical_coefficients(gr,HSr)
    @test norm(HSf[1][1] - HSr[2][1]) < 1.0e-14
    @test norm(HSf[2][1] - HSr[1][1]) < 1.0e-14
    @test norm(HSf[1][2] - HSr[2][2]) < 1.0e-14
    @test norm(HSf[2][2] - HSr[1][2]) < 1.0e-14
    @test norm(spdata_f1[1] - spdata_r2[1]) < 1.0e-14
    @test norm(spdata_f1[2] - spdata_r2[2]) < 1.0e-14
    @test norm(spdata_f1[3] - spdata_r2[3]) < 1.0e-14
    @test norm(spdata_f2[1] - spdata_r1[1]) < 1.0e-14
    @test norm(spdata_f2[2] - spdata_r1[2]) < 1.0e-14
    @test norm(spdata_f2[3] - spdata_r1[3]) < 1.0e-14
    @test abs(coeffs_f[1][1] - coeffs_r[2][1]) < 1.0e-14
    @test abs(coeffs_f[1][2] - coeffs_r[2][2]) < 1.0e-14
    @test abs(coeffs_f[2][1] - coeffs_r[1][1]) < 1.0e-14
    @test abs(coeffs_f[2][2] - coeffs_r[1][2]) < 1.0e-14
    gwr = SimpleWeightedDiGraph(transpose(weights(gw)))
    HSwf = hierarchical_structure(gw)
    HSwr = hierarchical_structure(gwr)
    spdata_wf1 = findnz(HSwf[1][3])
    spdata_wf2 = findnz(HSwf[2][3])
    coeffs_wf = hierarchical_coefficients(gw,HSwf)
    spdata_wr1 = findnz(HSwr[1][3])
    spdata_wr2 = findnz(HSwr[2][3])
    coeffs_wr = hierarchical_coefficients(gwr,HSwr)
    @test norm(HSwf[1][1] - HSwr[2][1]) < 1.0e-14
    @test norm(HSwf[2][1] - HSwr[1][1]) < 1.0e-14
    @test norm(HSwf[1][2] - HSwr[2][2]) < 1.0e-14
    @test norm(HSwf[2][2] - HSwr[1][2]) < 1.0e-14
    @test norm(spdata_wf1[1] - spdata_wr2[1]) < 1.0e-14
    @test norm(spdata_wf1[2] - spdata_wr2[2]) < 1.0e-14
    @test norm(spdata_wf1[3] - spdata_wr2[3]) < 1.0e-14
    @test norm(spdata_wf2[1] - spdata_wr1[1]) < 1.0e-14
    @test norm(spdata_wf2[2] - spdata_wr1[2]) < 1.0e-14
    @test norm(spdata_wf2[3] - spdata_wr1[3]) < 1.0e-14
    @test abs(coeffs_wf[1][1] - coeffs_wr[2][1]) < 1.0e-14
    @test abs(coeffs_wf[1][2] - coeffs_wr[2][2]) < 1.0e-14
    @test abs(coeffs_wf[2][1] - coeffs_wr[1][1]) < 1.0e-14
    @test abs(coeffs_wf[2][2] - coeffs_wr[1][2]) < 1.0e-14
    @test norm(HSf[1][1] - HSwr[2][1]) < 1.0e-14
    @test norm(HSf[2][1] - HSwr[1][1]) < 1.0e-14
    @test norm(HSf[1][2] - HSwr[2][2]) < 1.0e-14
    @test norm(HSf[2][2] - HSwr[1][2]) < 1.0e-14
    @test norm(spdata_f1[1] - spdata_wr2[1]) < 1.0e-14
    @test norm(spdata_f1[2] - spdata_wr2[2]) < 1.0e-14
    @test norm(spdata_f1[3] - spdata_wr2[3]) < 1.0e-14
    @test norm(spdata_f2[1] - spdata_wr1[1]) < 1.0e-14
    @test norm(spdata_f2[2] - spdata_wr1[2]) < 1.0e-14
    @test norm(spdata_f2[3] - spdata_wr1[3]) < 1.0e-14
    @test abs(coeffs_f[1][1] - coeffs_wr[2][1]) < 1.0e-14
    @test abs(coeffs_f[1][2] - coeffs_wr[2][2]) < 1.0e-14
    @test abs(coeffs_f[2][1] - coeffs_wr[1][1]) < 1.0e-14
    @test abs(coeffs_f[2][2] - coeffs_wr[1][2]) < 1.0e-14

    B_HS = forward_hierarchical_structure(g, big=true)
    B_spdata = findnz(B_HS[3])
    B_coeffs = hierarchical_coefficients(g,B_HS)
    @test norm(B_HS[1] - [ceil(i/2) - big(36)/11 for i in 1:11]) < 1.0e-70
    @test norm(B_HS[2] - [BigFloat(1),BigFloat(1),BigFloat(0),BigFloat(0),BigFloat(0),BigFloat(0),BigFloat(0),BigFloat(0),BigFloat(0),BigFloat(0),BigFloat(0)]) < 1.0e-70
    @test norm(B_spdata[3] - [BigFloat(1) for i in 1:length(spdata[3])]) < 1.0e-70
    @test norm(B_HS[1] - HS[1]) < 1.0e-14
    @test norm(B_HS[2] - HS[2]) < 1.0e-14
    @test norm(B_spdata[1] - spdata[1]) < 1.0e-14
    @test norm(B_spdata[2] - spdata[2]) < 1.0e-14
    @test norm(B_spdata[3] - spdata[3]) < 1.0e-14
    @test abs(B_coeffs[1]) < 1.0e-70
    @test abs(B_coeffs[2]) < 1.0e-70
    B_HSA = forward_hierarchical_structure(adjacency_matrix(g), big=true)
    B_spdataA = findnz(B_HSA[3])
    B_coeffsA = hierarchical_coefficients(g,B_HSA)
    @test norm(B_HSA[1] - B_HS[1]) < 1.0e-70
    @test norm(B_HSA[2] - B_HS[2]) < 1.0e-70
    @test norm(B_spdataA[1] - B_spdata[1]) < 1.0e-70
    @test norm(B_spdataA[2] - B_spdata[2]) < 1.0e-70
    @test norm(B_spdataA[3] - B_spdata[3]) < 1.0e-70
    @test abs(B_coeffsA[1] - B_coeffs[1]) < 1.0e-70
    @test abs(B_coeffsA[2] - B_coeffs[2]) < 1.0e-70
    B_HSf = hierarchical_structure(g, big=true)
    B_HSr = hierarchical_structure(gr, big=true)
    B_spdata_f1 = findnz(B_HSf[1][3])
    B_spdata_f2 = findnz(B_HSf[2][3])
    B_coeffs_f = hierarchical_coefficients(g,B_HSf)
    B_spdata_r1 = findnz(B_HSr[1][3])
    B_spdata_r2 = findnz(B_HSr[2][3])
    B_coeffs_r = hierarchical_coefficients(gr,B_HSr)
    @test norm(B_HSf[1][1] - B_HSr[2][1]) < 1.0e-70
    @test norm(B_HSf[2][1] - B_HSr[1][1]) < 1.0e-70
    @test norm(B_HSf[1][2] - B_HSr[2][2]) < 1.0e-70
    @test norm(B_HSf[2][2] - B_HSr[1][2]) < 1.0e-70
    @test norm(B_spdata_f1[1] - B_spdata_r2[1]) < 1.0e-70
    @test norm(B_spdata_f1[2] - B_spdata_r2[2]) < 1.0e-70
    @test norm(B_spdata_f1[3] - B_spdata_r2[3]) < 1.0e-70
    @test norm(B_spdata_f2[1] - B_spdata_r1[1]) < 1.0e-70
    @test norm(B_spdata_f2[2] - B_spdata_r1[2]) < 1.0e-70
    @test norm(B_spdata_f2[3] - B_spdata_r1[3]) < 1.0e-70
    @test abs(B_coeffs_f[1][1] - B_coeffs_r[2][1]) < 1.0e-70
    @test abs(B_coeffs_f[1][2] - B_coeffs_r[2][2]) < 1.0e-70
    @test abs(B_coeffs_f[2][1] - B_coeffs_r[1][1]) < 1.0e-70
    @test abs(B_coeffs_f[2][2] - B_coeffs_r[1][2]) < 1.0e-70
    B_HSwf = hierarchical_structure(gw, big=true)
    B_HSwr = hierarchical_structure(gwr, big=true)
    B_spdata_wf1 = findnz(B_HSwf[1][3])
    B_spdata_wf2 = findnz(B_HSwf[2][3])
    B_coeffs_wf = hierarchical_coefficients(gw,B_HSwf)
    B_spdata_wr1 = findnz(B_HSwr[1][3])
    B_spdata_wr2 = findnz(B_HSwr[2][3])
    B_coeffs_wr = hierarchical_coefficients(gwr,B_HSwr)
    @test norm(B_HSwf[1][1] - B_HSwr[2][1]) < 1.0e-70
    @test norm(B_HSwf[2][1] - B_HSwr[1][1]) < 1.0e-70
    @test norm(B_HSwf[1][2] - B_HSwr[2][2]) < 1.0e-70
    @test norm(B_HSwf[2][2] - B_HSwr[1][2]) < 1.0e-70
    @test norm(B_spdata_wf1[1] - B_spdata_wr2[1]) < 1.0e-70
    @test norm(B_spdata_wf1[2] - B_spdata_wr2[2]) < 1.0e-70
    @test norm(B_spdata_wf1[3] - B_spdata_wr2[3]) < 1.0e-70
    @test norm(B_spdata_wf2[1] - B_spdata_wr1[1]) < 1.0e-70
    @test norm(B_spdata_wf2[2] - B_spdata_wr1[2]) < 1.0e-70
    @test norm(B_spdata_wf2[3] - B_spdata_wr1[3]) < 1.0e-70
    @test abs(B_coeffs_wf[1][1] - B_coeffs_wr[2][1]) < 1.0e-70
    @test abs(B_coeffs_wf[1][2] - B_coeffs_wr[2][2]) < 1.0e-70
    @test abs(B_coeffs_wf[2][1] - B_coeffs_wr[1][1]) < 1.0e-70
    @test abs(B_coeffs_wf[2][2] - B_coeffs_wr[1][2]) < 1.0e-70
    @test norm(B_HSf[1][1] - B_HSwr[2][1]) < 1.0e-70
    @test norm(B_HSf[2][1] - B_HSwr[1][1]) < 1.0e-70
    @test norm(B_HSf[1][2] - B_HSwr[2][2]) < 1.0e-70
    @test norm(B_HSf[2][2] - B_HSwr[1][2]) < 1.0e-70
    @test norm(B_spdata_f1[1] - B_spdata_wr2[1]) < 1.0e-70
    @test norm(B_spdata_f1[2] - B_spdata_wr2[2]) < 1.0e-70
    @test norm(B_spdata_f1[3] - B_spdata_wr2[3]) < 1.0e-70
    @test norm(B_spdata_f2[1] - B_spdata_wr1[1]) < 1.0e-70
    @test norm(B_spdata_f2[2] - B_spdata_wr1[2]) < 1.0e-70
    @test norm(B_spdata_f2[3] - B_spdata_wr1[3]) < 1.0e-70
    @test abs(B_coeffs_f[1][1] - B_coeffs_wr[2][1]) < 1.0e-70
    @test abs(B_coeffs_f[1][2] - B_coeffs_wr[2][2]) < 1.0e-70
    @test abs(B_coeffs_f[2][1] - B_coeffs_wr[1][1]) < 1.0e-70
    @test abs(B_coeffs_f[2][2] - B_coeffs_wr[1][2]) < 1.0e-70


    gw = SimpleWeightedDiGraph(10)
    add_edge!(gw,1,2,2)
    add_edge!(gw,2,3,2)
    add_edge!(gw,3,1,2)
    add_edge!(gw,1,4,2)
    add_edge!(gw,3,5,2)
    add_edge!(gw,4,6,2)
    add_edge!(gw,6,5,2)
    add_edge!(gw,10,8,2)
    add_edge!(gw,9,10,2)
    add_edge!(gw,8,4,2)
    add_edge!(gw,7,10,2)
    add_edge!(gw,4,7,2)
    add_edge!(gw,6,9,2)
    add_edge!(gw,5,6,2)
    g = SimpleDiGraph(adjacency_matrix(gw))
    HS = forward_hierarchical_structure(g)
    spdata = findnz(HS[3])
    coeffs = hierarchical_coefficients(g,HS)
    @test norm(HS[1] - [-4.24285714285708, -4.242857142857171, -4.242857142856976,  0.8999999999996752, -0.5285714285718714,  1.1857142857147973,  1.9000000000000274,  4.042857142857218,  2.185714285714233,  3.042857142857145]) < 1.0e-10
    @test norm(HS[2] - [1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]) < 1.0e-10
    @test norm(spdata[1] - [3,1,2,1,8,3,6,4,5,4,10,6,7,9]) < 1.0e-14
    @test norm(spdata[2] -[1,2,3,4,4,5,5,6,6,7,8,9,10,10]) < 1.0e-14
    @test norm(spdata[3] - [0,0,0,5.142857142856755,-3.142857142857543,3.7142857142851042,-1.7142857142866688,0.28571428571512214,1.7142857142866688,1,1,1,1.1428571428571175,0.8571428571429118]) < 1.0e-10
    @test abs(coeffs[1]-3.0/14.0) < 1.0e-10
    @test abs(coeffs[2]-1.936115253297853) < 1.0e-10
    HSA = forward_hierarchical_structure(adjacency_matrix(g))
    spdataA = findnz(HSA[3])
    coeffsA = hierarchical_coefficients(g,HSA)
    @test norm(HSA[1] - HS[1]) < 1.0e-10
    @test norm(HSA[2] - HS[2]) < 1.0e-10
    @test norm(spdataA[1] - spdata[1]) < 1.0e-14
    @test norm(spdataA[2] - spdata[2]) < 1.0e-14
    @test norm(spdataA[3] - spdata[3]) < 1.0e-10
    @test abs(coeffsA[1] - coeffs[1]) < 1.0e-10
    @test abs(coeffsA[2] - coeffs[2]) < 1.0e-10
    gr = SimpleDiGraph(transpose(adjacency_matrix(g)))
    HSf = hierarchical_structure(g)
    HSr = hierarchical_structure(gr)
    spdata_f1 = findnz(HSf[1][3])
    spdata_f2 = findnz(HSf[2][3])
    coeffs_f = hierarchical_coefficients(g,HSf)
    spdata_r1 = findnz(HSr[1][3])
    spdata_r2 = findnz(HSr[2][3])
    coeffs_r = hierarchical_coefficients(gr,HSr)
    @test norm(HSf[1][1] - HSr[2][1]) < 1.0e-14
    @test norm(HSf[2][1] - HSr[1][1]) < 1.0e-14
    @test norm(HSf[1][2] - HSr[2][2]) < 1.0e-14
    @test norm(HSf[2][2] - HSr[1][2]) < 1.0e-14
    @test norm(spdata_f1[1] - spdata_r2[1]) < 1.0e-14
    @test norm(spdata_f1[2] - spdata_r2[2]) < 1.0e-14
    @test norm(spdata_f1[3] - spdata_r2[3]) < 1.0e-14
    @test norm(spdata_f2[1] - spdata_r1[1]) < 1.0e-14
    @test norm(spdata_f2[2] - spdata_r1[2]) < 1.0e-14
    @test norm(spdata_f2[3] - spdata_r1[3]) < 1.0e-14
    @test abs(coeffs_f[1][1] - coeffs_r[2][1]) < 1.0e-14
    @test abs(coeffs_f[1][2] - coeffs_r[2][2]) < 1.0e-14
    @test abs(coeffs_f[2][1] - coeffs_r[1][1]) < 1.0e-14
    @test abs(coeffs_f[2][2] - coeffs_r[1][2]) < 1.0e-14
    gwr = SimpleWeightedDiGraph(transpose(weights(gw)))
    HSwf = hierarchical_structure(gw)
    HSwr = hierarchical_structure(gwr)
    spdata_wf1 = findnz(HSwf[1][3])
    spdata_wf2 = findnz(HSwf[2][3])
    coeffs_wf = hierarchical_coefficients(gw,HSwf)
    spdata_wr1 = findnz(HSwr[1][3])
    spdata_wr2 = findnz(HSwr[2][3])
    coeffs_wr = hierarchical_coefficients(gwr,HSwr)
    @test norm(HSwf[1][1] - HSwr[2][1]) < 1.0e-14
    @test norm(HSwf[2][1] - HSwr[1][1]) < 1.0e-14
    @test norm(HSwf[1][2] - HSwr[2][2]) < 1.0e-14
    @test norm(HSwf[2][2] - HSwr[1][2]) < 1.0e-14
    @test norm(spdata_wf1[1] - spdata_wr2[1]) < 1.0e-14
    @test norm(spdata_wf1[2] - spdata_wr2[2]) < 1.0e-14
    @test norm(spdata_wf1[3] - spdata_wr2[3]) < 1.0e-14
    @test norm(spdata_wf2[1] - spdata_wr1[1]) < 1.0e-14
    @test norm(spdata_wf2[2] - spdata_wr1[2]) < 1.0e-14
    @test norm(spdata_wf2[3] - spdata_wr1[3]) < 1.0e-14
    @test abs(coeffs_wf[1][1] - coeffs_wr[2][1]) < 1.0e-14
    @test abs(coeffs_wf[1][2] - coeffs_wr[2][2]) < 1.0e-14
    @test abs(coeffs_wf[2][1] - coeffs_wr[1][1]) < 1.0e-14
    @test abs(coeffs_wf[2][2] - coeffs_wr[1][2]) < 1.0e-14
    @test norm(HSf[1][1] - HSwr[2][1]) < 1.0e-14
    @test norm(HSf[2][1] - HSwr[1][1]) < 1.0e-14
    @test norm(HSf[1][2] - HSwr[2][2]) < 1.0e-14
    @test norm(HSf[2][2] - HSwr[1][2]) < 1.0e-14
    @test norm(spdata_f1[1] - spdata_wr2[1]) < 1.0e-14
    @test norm(spdata_f1[2] - spdata_wr2[2]) < 1.0e-14
    @test norm(spdata_f1[3] - spdata_wr2[3]) < 1.0e-14
    @test norm(spdata_f2[1] - spdata_wr1[1]) < 1.0e-14
    @test norm(spdata_f2[2] - spdata_wr1[2]) < 1.0e-14
    @test norm(spdata_f2[3] - spdata_wr1[3]) < 1.0e-14
    @test abs(coeffs_f[1][1] - coeffs_wr[2][1]) < 1.0e-14
    @test abs(coeffs_f[1][2] - coeffs_wr[2][2]) < 1.0e-14
    @test abs(coeffs_f[2][1] - coeffs_wr[1][1]) < 1.0e-14
    @test abs(coeffs_f[2][2] - coeffs_wr[1][2]) < 1.0e-14

    B_HS = forward_hierarchical_structure(g, big=true)
    B_spdata = findnz(B_HS[3])
    B_coeffs = hierarchical_coefficients(g,B_HS)
    @test norm(B_HS[2] - [BigFloat(1),BigFloat(1),BigFloat(1),BigFloat(0),BigFloat(0),BigFloat(0),BigFloat(0),BigFloat(0),BigFloat(0),BigFloat(0)]) < 1.0e-70
    @test norm(B_HS[1] - HS[1]) < 1.0e-10
    @test norm(B_HS[2] - HS[2]) < 1.0e-10
    @test norm(B_spdata[1] - spdata[1]) < 1.0e-14
    @test norm(B_spdata[2] - spdata[2]) < 1.0e-14
    @test norm(B_spdata[3] - spdata[3]) < 1.0e-10
    @test abs(B_coeffs[1] - big(3)/14) < 1.0e-70
    @test abs(B_coeffs[2] - 1.936115253297853) < 1.0e-10
    B_HSA = forward_hierarchical_structure(adjacency_matrix(g), big=true)
    B_spdataA = findnz(B_HSA[3])
    B_coeffsA = hierarchical_coefficients(g,B_HSA)
    @test norm(B_HSA[1] - B_HS[1]) < 1.0e-70
    @test norm(B_HSA[2] - B_HS[2]) < 1.0e-70
    @test norm(B_spdataA[1] - B_spdata[1]) < 1.0e-70
    @test norm(B_spdataA[2] - B_spdata[2]) < 1.0e-70
    @test norm(B_spdataA[3] - B_spdata[3]) < 1.0e-70
    @test abs(B_coeffsA[1] - B_coeffs[1]) < 1.0e-70
    @test abs(B_coeffsA[2] - B_coeffs[2]) < 1.0e-70
    B_HSf = hierarchical_structure(g, big=true)
    B_HSr = hierarchical_structure(gr, big=true)
    B_spdata_f1 = findnz(B_HSf[1][3])
    B_spdata_f2 = findnz(B_HSf[2][3])
    B_coeffs_f = hierarchical_coefficients(g,B_HSf)
    B_spdata_r1 = findnz(B_HSr[1][3])
    B_spdata_r2 = findnz(B_HSr[2][3])
    B_coeffs_r = hierarchical_coefficients(gr,B_HSr)
    @test norm(B_HSf[1][1] - B_HSr[2][1]) < 1.0e-70
    @test norm(B_HSf[2][1] - B_HSr[1][1]) < 1.0e-70
    @test norm(B_HSf[1][2] - B_HSr[2][2]) < 1.0e-70
    @test norm(B_HSf[2][2] - B_HSr[1][2]) < 1.0e-70
    @test norm(B_spdata_f1[1] - B_spdata_r2[1]) < 1.0e-70
    @test norm(B_spdata_f1[2] - B_spdata_r2[2]) < 1.0e-70
    @test norm(B_spdata_f1[3] - B_spdata_r2[3]) < 1.0e-70
    @test norm(B_spdata_f2[1] - B_spdata_r1[1]) < 1.0e-70
    @test norm(B_spdata_f2[2] - B_spdata_r1[2]) < 1.0e-70
    @test norm(B_spdata_f2[3] - B_spdata_r1[3]) < 1.0e-70
    @test abs(B_coeffs_f[1][1] - B_coeffs_r[2][1]) < 1.0e-70
    @test abs(B_coeffs_f[1][2] - B_coeffs_r[2][2]) < 1.0e-70
    @test abs(B_coeffs_f[2][1] - B_coeffs_r[1][1]) < 1.0e-70
    @test abs(B_coeffs_f[2][2] - B_coeffs_r[1][2]) < 1.0e-70
    B_HSwf = hierarchical_structure(gw, big=true)
    B_HSwr = hierarchical_structure(gwr, big=true)
    B_spdata_wf1 = findnz(B_HSwf[1][3])
    B_spdata_wf2 = findnz(B_HSwf[2][3])
    B_coeffs_wf = hierarchical_coefficients(gw,B_HSwf)
    B_spdata_wr1 = findnz(B_HSwr[1][3])
    B_spdata_wr2 = findnz(B_HSwr[2][3])
    B_coeffs_wr = hierarchical_coefficients(gwr,B_HSwr)
    @test norm(B_HSwf[1][1] - B_HSwr[2][1]) < 1.0e-70
    @test norm(B_HSwf[2][1] - B_HSwr[1][1]) < 1.0e-70
    @test norm(B_HSwf[1][2] - B_HSwr[2][2]) < 1.0e-70
    @test norm(B_HSwf[2][2] - B_HSwr[1][2]) < 1.0e-70
    @test norm(B_spdata_wf1[1] - B_spdata_wr2[1]) < 1.0e-70
    @test norm(B_spdata_wf1[2] - B_spdata_wr2[2]) < 1.0e-70
    @test norm(B_spdata_wf1[3] - B_spdata_wr2[3]) < 1.0e-70
    @test norm(B_spdata_wf2[1] - B_spdata_wr1[1]) < 1.0e-70
    @test norm(B_spdata_wf2[2] - B_spdata_wr1[2]) < 1.0e-70
    @test norm(B_spdata_wf2[3] - B_spdata_wr1[3]) < 1.0e-70
    @test abs(B_coeffs_wf[1][1] - B_coeffs_wr[2][1]) < 1.0e-70
    @test abs(B_coeffs_wf[1][2] - B_coeffs_wr[2][2]) < 1.0e-70
    @test abs(B_coeffs_wf[2][1] - B_coeffs_wr[1][1]) < 1.0e-70
    @test abs(B_coeffs_wf[2][2] - B_coeffs_wr[1][2]) < 1.0e-70
    @test norm(B_HSf[1][1] - B_HSwr[2][1]) < 1.0e-70
    @test norm(B_HSf[2][1] - B_HSwr[1][1]) < 1.0e-70
    @test norm(B_HSf[1][2] - B_HSwr[2][2]) < 1.0e-70
    @test norm(B_HSf[2][2] - B_HSwr[1][2]) < 1.0e-70
    @test norm(B_spdata_f1[1] - B_spdata_wr2[1]) < 1.0e-70
    @test norm(B_spdata_f1[2] - B_spdata_wr2[2]) < 1.0e-70
    @test norm(B_spdata_f1[3] - B_spdata_wr2[3]) < 1.0e-70
    @test norm(B_spdata_f2[1] - B_spdata_wr1[1]) < 1.0e-70
    @test norm(B_spdata_f2[2] - B_spdata_wr1[2]) < 1.0e-70
    @test norm(B_spdata_f2[3] - B_spdata_wr1[3]) < 1.0e-70
    @test abs(B_coeffs_f[1][1] - B_coeffs_wr[2][1]) < 1.0e-70
    @test abs(B_coeffs_f[1][2] - B_coeffs_wr[2][2]) < 1.0e-70
    @test abs(B_coeffs_f[2][1] - B_coeffs_wr[1][1]) < 1.0e-70
    @test abs(B_coeffs_f[2][2] - B_coeffs_wr[1][2]) < 1.0e-70


    gw = SimpleWeightedDiGraph(6)
    add_edge!(gw,1,2,2)
    add_edge!(gw,2,3,2)
    add_edge!(gw,3,4,2)
    add_edge!(gw,4,5,2)
    add_edge!(gw,5,6,2)
    add_edge!(gw,6,1,2)
    g = SimpleDiGraph(adjacency_matrix(gw))
    HS = forward_hierarchical_structure(g)
    spdata = findnz(HS[3])
    coeffs = hierarchical_coefficients(g,HS)
    @test norm(HS[1]) < 1.0e-10
    @test norm(HS[2] - [1.0,1.0,1.0,1.0,1.0,1.0]) < 1.0e-10
    @test norm(spdata[1] - [6,1,2,3,4,5]) < 1.0e-14
    @test norm(spdata[2] - [1,2,3,4,5,6]) < 1.0e-14
    @test norm(spdata[3]) < 1.0e-10
    @test abs(coeffs[1]-1.0) < 1.0e-10
    @test abs(coeffs[2]-0.0) < 1.0e-10
    HSA = forward_hierarchical_structure(adjacency_matrix(g))
    spdataA = findnz(HSA[3])
    coeffsA = hierarchical_coefficients(g,HSA)
    @test norm(HSA[1] - HS[1]) < 1.0e-10
    @test norm(HSA[2] - HS[2]) < 1.0e-10
    @test norm(spdataA[1] - spdata[1]) < 1.0e-14
    @test norm(spdataA[2] - spdata[2]) < 1.0e-14
    @test norm(spdataA[3] - spdata[3]) < 1.0e-10
    @test abs(coeffsA[1] - coeffs[1]) < 1.0e-10
    @test abs(coeffsA[2] - coeffs[2]) < 1.0e-10
    gr = SimpleDiGraph(transpose(adjacency_matrix(g)))
    HSf = hierarchical_structure(g)
    HSr = hierarchical_structure(gr)
    spdata_f1 = findnz(HSf[1][3])
    spdata_f2 = findnz(HSf[2][3])
    coeffs_f = hierarchical_coefficients(g,HSf)
    spdata_r1 = findnz(HSr[1][3])
    spdata_r2 = findnz(HSr[2][3])
    coeffs_r = hierarchical_coefficients(gr,HSr)
    @test norm(HSf[1][1] - HSr[2][1]) < 1.0e-14
    @test norm(HSf[2][1] - HSr[1][1]) < 1.0e-14
    @test norm(HSf[1][2] - HSr[2][2]) < 1.0e-14
    @test norm(HSf[2][2] - HSr[1][2]) < 1.0e-14
    @test norm(spdata_f1[1] - spdata_r2[1]) < 1.0e-14
    @test norm(spdata_f1[2] - spdata_r2[2]) < 1.0e-14
    @test norm(spdata_f1[3] - spdata_r2[3]) < 1.0e-14
    @test norm(spdata_f2[1] - spdata_r1[1]) < 1.0e-14
    @test norm(spdata_f2[2] - spdata_r1[2]) < 1.0e-14
    @test norm(spdata_f2[3] - spdata_r1[3]) < 1.0e-14
    @test abs(coeffs_f[1][1] - coeffs_r[2][1]) < 1.0e-14
    @test abs(coeffs_f[1][2] - coeffs_r[2][2]) < 1.0e-14
    @test abs(coeffs_f[2][1] - coeffs_r[1][1]) < 1.0e-14
    @test abs(coeffs_f[2][2] - coeffs_r[1][2]) < 1.0e-14
    gwr = SimpleWeightedDiGraph(transpose(weights(gw)))
    HSwf = hierarchical_structure(gw)
    HSwr = hierarchical_structure(gwr)
    spdata_wf1 = findnz(HSwf[1][3])
    spdata_wf2 = findnz(HSwf[2][3])
    coeffs_wf = hierarchical_coefficients(gw,HSwf)
    spdata_wr1 = findnz(HSwr[1][3])
    spdata_wr2 = findnz(HSwr[2][3])
    coeffs_wr = hierarchical_coefficients(gwr,HSwr)
    @test norm(HSwf[1][1] - HSwr[2][1]) < 1.0e-14
    @test norm(HSwf[2][1] - HSwr[1][1]) < 1.0e-14
    @test norm(HSwf[1][2] - HSwr[2][2]) < 1.0e-14
    @test norm(HSwf[2][2] - HSwr[1][2]) < 1.0e-14
    @test norm(spdata_wf1[1] - spdata_wr2[1]) < 1.0e-14
    @test norm(spdata_wf1[2] - spdata_wr2[2]) < 1.0e-14
    @test norm(spdata_wf1[3] - spdata_wr2[3]) < 1.0e-14
    @test norm(spdata_wf2[1] - spdata_wr1[1]) < 1.0e-14
    @test norm(spdata_wf2[2] - spdata_wr1[2]) < 1.0e-14
    @test norm(spdata_wf2[3] - spdata_wr1[3]) < 1.0e-14
    @test abs(coeffs_wf[1][1] - coeffs_wr[2][1]) < 1.0e-14
    @test abs(coeffs_wf[1][2] - coeffs_wr[2][2]) < 1.0e-14
    @test abs(coeffs_wf[2][1] - coeffs_wr[1][1]) < 1.0e-14
    @test abs(coeffs_wf[2][2] - coeffs_wr[1][2]) < 1.0e-14
    @test norm(HSf[1][1] - HSwr[2][1]) < 1.0e-14
    @test norm(HSf[2][1] - HSwr[1][1]) < 1.0e-14
    @test norm(HSf[1][2] - HSwr[2][2]) < 1.0e-14
    @test norm(HSf[2][2] - HSwr[1][2]) < 1.0e-14
    @test norm(spdata_f1[1] - spdata_wr2[1]) < 1.0e-14
    @test norm(spdata_f1[2] - spdata_wr2[2]) < 1.0e-14
    @test norm(spdata_f1[3] - spdata_wr2[3]) < 1.0e-14
    @test norm(spdata_f2[1] - spdata_wr1[1]) < 1.0e-14
    @test norm(spdata_f2[2] - spdata_wr1[2]) < 1.0e-14
    @test norm(spdata_f2[3] - spdata_wr1[3]) < 1.0e-14
    @test abs(coeffs_f[1][1] - coeffs_wr[2][1]) < 1.0e-14
    @test abs(coeffs_f[1][2] - coeffs_wr[2][2]) < 1.0e-14
    @test abs(coeffs_f[2][1] - coeffs_wr[1][1]) < 1.0e-14
    @test abs(coeffs_f[2][2] - coeffs_wr[1][2]) < 1.0e-14

    B_HS = forward_hierarchical_structure(g, big=true)
    B_spdata = findnz(B_HS[3])
    B_coeffs = hierarchical_coefficients(g,B_HS)
    @test norm(B_HS[1]) < 1.0e-70
    @test norm(B_HS[2] - [BigFloat(1),BigFloat(1),BigFloat(1),BigFloat(1),BigFloat(1),BigFloat(1)]) < 1.0e-70
    @test norm(B_spdata[3]) < 1.0e-70
    @test norm(B_HS[1] - HS[1]) < 1.0e-14
    @test norm(B_HS[2] - HS[2]) < 1.0e-14
    @test norm(B_spdata[1] - spdata[1]) < 1.0e-14
    @test norm(B_spdata[2] - spdata[2]) < 1.0e-14
    @test norm(B_spdata[3] - spdata[3]) < 1.0e-14
    @test abs(B_coeffs[1] - 1) < 1.0e-70
    @test abs(B_coeffs[2]) < 1.0e-70
    B_HSA = forward_hierarchical_structure(adjacency_matrix(g), big=true)
    B_spdataA = findnz(B_HSA[3])
    B_coeffsA = hierarchical_coefficients(g,B_HSA)
    @test norm(B_HSA[1] - B_HS[1]) < 1.0e-70
    @test norm(B_HSA[2] - B_HS[2]) < 1.0e-70
    @test norm(B_spdataA[1] - B_spdata[1]) < 1.0e-70
    @test norm(B_spdataA[2] - B_spdata[2]) < 1.0e-70
    @test norm(B_spdataA[3] - B_spdata[3]) < 1.0e-70
    @test abs(B_coeffsA[1] - B_coeffs[1]) < 1.0e-70
    @test abs(B_coeffsA[2] - B_coeffs[2]) < 1.0e-70
    B_HSf = hierarchical_structure(g, big=true)
    B_HSr = hierarchical_structure(gr, big=true)
    B_spdata_f1 = findnz(B_HSf[1][3])
    B_spdata_f2 = findnz(B_HSf[2][3])
    B_coeffs_f = hierarchical_coefficients(g,B_HSf)
    B_spdata_r1 = findnz(B_HSr[1][3])
    B_spdata_r2 = findnz(B_HSr[2][3])
    B_coeffs_r = hierarchical_coefficients(gr,B_HSr)
    @test norm(B_HSf[1][1] - B_HSr[2][1]) < 1.0e-70
    @test norm(B_HSf[2][1] - B_HSr[1][1]) < 1.0e-70
    @test norm(B_HSf[1][2] - B_HSr[2][2]) < 1.0e-70
    @test norm(B_HSf[2][2] - B_HSr[1][2]) < 1.0e-70
    @test norm(B_spdata_f1[1] - B_spdata_r2[1]) < 1.0e-70
    @test norm(B_spdata_f1[2] - B_spdata_r2[2]) < 1.0e-70
    @test norm(B_spdata_f1[3] - B_spdata_r2[3]) < 1.0e-70
    @test norm(B_spdata_f2[1] - B_spdata_r1[1]) < 1.0e-70
    @test norm(B_spdata_f2[2] - B_spdata_r1[2]) < 1.0e-70
    @test norm(B_spdata_f2[3] - B_spdata_r1[3]) < 1.0e-70
    @test abs(B_coeffs_f[1][1] - B_coeffs_r[2][1]) < 1.0e-70
    @test abs(B_coeffs_f[1][2] - B_coeffs_r[2][2]) < 1.0e-70
    @test abs(B_coeffs_f[2][1] - B_coeffs_r[1][1]) < 1.0e-70
    @test abs(B_coeffs_f[2][2] - B_coeffs_r[1][2]) < 1.0e-70
    B_HSwf = hierarchical_structure(gw, big=true)
    B_HSwr = hierarchical_structure(gwr, big=true)
    B_spdata_wf1 = findnz(B_HSwf[1][3])
    B_spdata_wf2 = findnz(B_HSwf[2][3])
    B_coeffs_wf = hierarchical_coefficients(gw,B_HSwf)
    B_spdata_wr1 = findnz(B_HSwr[1][3])
    B_spdata_wr2 = findnz(B_HSwr[2][3])
    B_coeffs_wr = hierarchical_coefficients(gwr,B_HSwr)
    @test norm(B_HSwf[1][1] - B_HSwr[2][1]) < 1.0e-70
    @test norm(B_HSwf[2][1] - B_HSwr[1][1]) < 1.0e-70
    @test norm(B_HSwf[1][2] - B_HSwr[2][2]) < 1.0e-70
    @test norm(B_HSwf[2][2] - B_HSwr[1][2]) < 1.0e-70
    @test norm(B_spdata_wf1[1] - B_spdata_wr2[1]) < 1.0e-70
    @test norm(B_spdata_wf1[2] - B_spdata_wr2[2]) < 1.0e-70
    @test norm(B_spdata_wf1[3] - B_spdata_wr2[3]) < 1.0e-70
    @test norm(B_spdata_wf2[1] - B_spdata_wr1[1]) < 1.0e-70
    @test norm(B_spdata_wf2[2] - B_spdata_wr1[2]) < 1.0e-70
    @test norm(B_spdata_wf2[3] - B_spdata_wr1[3]) < 1.0e-70
    @test abs(B_coeffs_wf[1][1] - B_coeffs_wr[2][1]) < 1.0e-70
    @test abs(B_coeffs_wf[1][2] - B_coeffs_wr[2][2]) < 1.0e-70
    @test abs(B_coeffs_wf[2][1] - B_coeffs_wr[1][1]) < 1.0e-70
    @test abs(B_coeffs_wf[2][2] - B_coeffs_wr[1][2]) < 1.0e-70
    @test norm(B_HSf[1][1] - B_HSwr[2][1]) < 1.0e-70
    @test norm(B_HSf[2][1] - B_HSwr[1][1]) < 1.0e-70
    @test norm(B_HSf[1][2] - B_HSwr[2][2]) < 1.0e-70
    @test norm(B_HSf[2][2] - B_HSwr[1][2]) < 1.0e-70
    @test norm(B_spdata_f1[1] - B_spdata_wr2[1]) < 1.0e-70
    @test norm(B_spdata_f1[2] - B_spdata_wr2[2]) < 1.0e-70
    @test norm(B_spdata_f1[3] - B_spdata_wr2[3]) < 1.0e-70
    @test norm(B_spdata_f2[1] - B_spdata_wr1[1]) < 1.0e-70
    @test norm(B_spdata_f2[2] - B_spdata_wr1[2]) < 1.0e-70
    @test norm(B_spdata_f2[3] - B_spdata_wr1[3]) < 1.0e-70
    @test abs(B_coeffs_f[1][1] - B_coeffs_wr[2][1]) < 1.0e-70
    @test abs(B_coeffs_f[1][2] - B_coeffs_wr[2][2]) < 1.0e-70
    @test abs(B_coeffs_f[2][1] - B_coeffs_wr[1][1]) < 1.0e-70
    @test abs(B_coeffs_f[2][2] - B_coeffs_wr[1][2]) < 1.0e-70


    gw = SimpleWeightedDiGraph(4)
    add_edge!(gw,1,2,2)
    add_edge!(gw,1,3,2)
    add_edge!(gw,1,4,2)
    add_edge!(gw,2,3,2)
    add_edge!(gw,3,4,2)
    add_edge!(gw,4,1,2)
    g = SimpleDiGraph(adjacency_matrix(gw))
    HS = forward_hierarchical_structure(g)
    spdata = findnz(HS[3])
    coeffs = hierarchical_coefficients(g,HS)
    @test norm(HS[1] - [-(5.0/8.0), -(1.0/8.0), 3.0/8.0, 3.0/8.0]) < 1.0e-14
    @test norm(HS[2] - [2.0,0.5,0.25,0.5]) < 1.0e-14
    @test norm(spdata[1] - [4,1,1,2,1,3]) < 1.0e-14
    @test norm(spdata[2] - [1,2,3,3,4,4]) < 1.0e-14
    @test norm(spdata[3] -[-1.0,0.5,1.0,0.5,1.0,0.0]) < 1.0e-14
    @test abs(coeffs[1] - 2.0/3.0) < 1.0e-14
    @test abs(coeffs[2] - 0.687184270936277) < 1.0e-14
    HSA = forward_hierarchical_structure(adjacency_matrix(g))
    spdataA = findnz(HSA[3])
    coeffsA = hierarchical_coefficients(g,HSA)
    @test norm(HSA[1] - HS[1]) < 1.0e-14
    @test norm(HSA[2] - HS[2]) < 1.0e-14
    @test norm(spdataA[1] - spdata[1]) < 1.0e-14
    @test norm(spdataA[2] - spdata[2]) < 1.0e-14
    @test norm(spdataA[3] - spdata[3]) < 1.0e-14
    @test abs(coeffsA[1] - coeffs[1]) < 1.0e-14
    @test abs(coeffsA[2] - coeffs[2]) < 1.0e-14
    gr = SimpleDiGraph(transpose(adjacency_matrix(g)))
    HSf = hierarchical_structure(g)
    HSr = hierarchical_structure(gr)
    spdata_f1 = findnz(HSf[1][3])
    spdata_f2 = findnz(HSf[2][3])
    coeffs_f = hierarchical_coefficients(g,HSf)
    spdata_r1 = findnz(HSr[1][3])
    spdata_r2 = findnz(HSr[2][3])
    coeffs_r = hierarchical_coefficients(gr,HSr)
    @test norm(HSf[1][1] - HSr[2][1]) < 1.0e-14
    @test norm(HSf[2][1] - HSr[1][1]) < 1.0e-14
    @test norm(HSf[1][2] - HSr[2][2]) < 1.0e-14
    @test norm(HSf[2][2] - HSr[1][2]) < 1.0e-14
    @test norm(spdata_f1[1] - spdata_r2[1]) < 1.0e-14
    @test norm(spdata_f1[2] - spdata_r2[2]) < 1.0e-14
    @test norm(spdata_f1[3] - spdata_r2[3]) < 1.0e-14
    @test norm(spdata_f2[1] - spdata_r1[1]) < 1.0e-14
    @test norm(spdata_f2[2] - spdata_r1[2]) < 1.0e-14
    @test norm(spdata_f2[3] - spdata_r1[3]) < 1.0e-14
    @test abs(coeffs_f[1][1] - coeffs_r[2][1]) < 1.0e-14
    @test abs(coeffs_f[1][2] - coeffs_r[2][2]) < 1.0e-14
    @test abs(coeffs_f[2][1] - coeffs_r[1][1]) < 1.0e-14
    @test abs(coeffs_f[2][2] - coeffs_r[1][2]) < 1.0e-14
    gwr = SimpleWeightedDiGraph(transpose(weights(gw)))
    HSwf = hierarchical_structure(gw)
    HSwr = hierarchical_structure(gwr)
    spdata_wf1 = findnz(HSwf[1][3])
    spdata_wf2 = findnz(HSwf[2][3])
    coeffs_wf = hierarchical_coefficients(gw,HSwf)
    spdata_wr1 = findnz(HSwr[1][3])
    spdata_wr2 = findnz(HSwr[2][3])
    coeffs_wr = hierarchical_coefficients(gwr,HSwr)
    @test norm(HSwf[1][1] - HSwr[2][1]) < 1.0e-14
    @test norm(HSwf[2][1] - HSwr[1][1]) < 1.0e-14
    @test norm(HSwf[1][2] - HSwr[2][2]) < 1.0e-14
    @test norm(HSwf[2][2] - HSwr[1][2]) < 1.0e-14
    @test norm(spdata_wf1[1] - spdata_wr2[1]) < 1.0e-14
    @test norm(spdata_wf1[2] - spdata_wr2[2]) < 1.0e-14
    @test norm(spdata_wf1[3] - spdata_wr2[3]) < 1.0e-14
    @test norm(spdata_wf2[1] - spdata_wr1[1]) < 1.0e-14
    @test norm(spdata_wf2[2] - spdata_wr1[2]) < 1.0e-14
    @test norm(spdata_wf2[3] - spdata_wr1[3]) < 1.0e-14
    @test abs(coeffs_wf[1][1] - coeffs_wr[2][1]) < 1.0e-14
    @test abs(coeffs_wf[1][2] - coeffs_wr[2][2]) < 1.0e-14
    @test abs(coeffs_wf[2][1] - coeffs_wr[1][1]) < 1.0e-14
    @test abs(coeffs_wf[2][2] - coeffs_wr[1][2]) < 1.0e-14
    @test norm(HSf[1][1] - HSwr[2][1]) < 1.0e-14
    @test norm(HSf[2][1] - HSwr[1][1]) < 1.0e-14
    @test norm(HSf[1][2] - HSwr[2][2]) < 1.0e-14
    @test norm(HSf[2][2] - HSwr[1][2]) < 1.0e-14
    @test norm(spdata_f1[1] - spdata_wr2[1]) < 1.0e-14
    @test norm(spdata_f1[2] - spdata_wr2[2]) < 1.0e-14
    @test norm(spdata_f1[3] - spdata_wr2[3]) < 1.0e-14
    @test norm(spdata_f2[1] - spdata_wr1[1]) < 1.0e-14
    @test norm(spdata_f2[2] - spdata_wr1[2]) < 1.0e-14
    @test norm(spdata_f2[3] - spdata_wr1[3]) < 1.0e-14
    @test abs(coeffs_f[1][1] - coeffs_wr[2][1]) < 1.0e-14
    @test abs(coeffs_f[1][2] - coeffs_wr[2][2]) < 1.0e-14
    @test abs(coeffs_f[2][1] - coeffs_wr[1][1]) < 1.0e-14
    @test abs(coeffs_f[2][2] - coeffs_wr[1][2]) < 1.0e-14

    B_HS = forward_hierarchical_structure(g, big=true)
    B_spdata = findnz(B_HS[3])
    B_coeffs = hierarchical_coefficients(g,B_HS)
    @test norm(B_HS[1] - HS[1]) < 1.0e-14
    @test norm(B_HS[2] - HS[2]) < 1.0e-14
    @test norm(B_spdata[1] - spdata[1]) < 1.0e-14
    @test norm(B_spdata[2] - spdata[2]) < 1.0e-14
    @test norm(B_spdata[3] - spdata[3]) < 1.0e-14
    @test abs(B_coeffs[1] - big(2)/3) < 1.0e-70
    @test abs(B_coeffs[2] - 0.687184270936277) < 1.0e-14
    B_HSA = forward_hierarchical_structure(adjacency_matrix(g), big=true)
    B_spdataA = findnz(B_HSA[3])
    B_coeffsA = hierarchical_coefficients(g,B_HSA)
    @test norm(B_HSA[1] - B_HS[1]) < 1.0e-70
    @test norm(B_HSA[2] - B_HS[2]) < 1.0e-70
    @test norm(B_spdataA[1] - B_spdata[1]) < 1.0e-70
    @test norm(B_spdataA[2] - B_spdata[2]) < 1.0e-70
    @test norm(B_spdataA[3] - B_spdata[3]) < 1.0e-70
    @test abs(B_coeffsA[1] - B_coeffs[1]) < 1.0e-70
    @test abs(B_coeffsA[2] - B_coeffs[2]) < 1.0e-70
    B_HSf = hierarchical_structure(g, big=true)
    B_HSr = hierarchical_structure(gr, big=true)
    B_spdata_f1 = findnz(B_HSf[1][3])
    B_spdata_f2 = findnz(B_HSf[2][3])
    B_coeffs_f = hierarchical_coefficients(g,B_HSf)
    B_spdata_r1 = findnz(B_HSr[1][3])
    B_spdata_r2 = findnz(B_HSr[2][3])
    B_coeffs_r = hierarchical_coefficients(gr,B_HSr)
    @test norm(B_HSf[1][1] - B_HSr[2][1]) < 1.0e-70
    @test norm(B_HSf[2][1] - B_HSr[1][1]) < 1.0e-70
    @test norm(B_HSf[1][2] - B_HSr[2][2]) < 1.0e-70
    @test norm(B_HSf[2][2] - B_HSr[1][2]) < 1.0e-70
    @test norm(B_spdata_f1[1] - B_spdata_r2[1]) < 1.0e-70
    @test norm(B_spdata_f1[2] - B_spdata_r2[2]) < 1.0e-70
    @test norm(B_spdata_f1[3] - B_spdata_r2[3]) < 1.0e-70
    @test norm(B_spdata_f2[1] - B_spdata_r1[1]) < 1.0e-70
    @test norm(B_spdata_f2[2] - B_spdata_r1[2]) < 1.0e-70
    @test norm(B_spdata_f2[3] - B_spdata_r1[3]) < 1.0e-70
    @test abs(B_coeffs_f[1][1] - B_coeffs_r[2][1]) < 1.0e-70
    @test abs(B_coeffs_f[1][2] - B_coeffs_r[2][2]) < 1.0e-70
    @test abs(B_coeffs_f[2][1] - B_coeffs_r[1][1]) < 1.0e-70
    @test abs(B_coeffs_f[2][2] - B_coeffs_r[1][2]) < 1.0e-70
    B_HSwf = hierarchical_structure(gw, big=true)
    B_HSwr = hierarchical_structure(gwr, big=true)
    B_spdata_wf1 = findnz(B_HSwf[1][3])
    B_spdata_wf2 = findnz(B_HSwf[2][3])
    B_coeffs_wf = hierarchical_coefficients(gw,B_HSwf)
    B_spdata_wr1 = findnz(B_HSwr[1][3])
    B_spdata_wr2 = findnz(B_HSwr[2][3])
    B_coeffs_wr = hierarchical_coefficients(gwr,B_HSwr)
    @test norm(B_HSwf[1][1] - B_HSwr[2][1]) < 1.0e-70
    @test norm(B_HSwf[2][1] - B_HSwr[1][1]) < 1.0e-70
    @test norm(B_HSwf[1][2] - B_HSwr[2][2]) < 1.0e-70
    @test norm(B_HSwf[2][2] - B_HSwr[1][2]) < 1.0e-70
    @test norm(B_spdata_wf1[1] - B_spdata_wr2[1]) < 1.0e-70
    @test norm(B_spdata_wf1[2] - B_spdata_wr2[2]) < 1.0e-70
    @test norm(B_spdata_wf1[3] - B_spdata_wr2[3]) < 1.0e-70
    @test norm(B_spdata_wf2[1] - B_spdata_wr1[1]) < 1.0e-70
    @test norm(B_spdata_wf2[2] - B_spdata_wr1[2]) < 1.0e-70
    @test norm(B_spdata_wf2[3] - B_spdata_wr1[3]) < 1.0e-70
    @test abs(B_coeffs_wf[1][1] - B_coeffs_wr[2][1]) < 1.0e-70
    @test abs(B_coeffs_wf[1][2] - B_coeffs_wr[2][2]) < 1.0e-70
    @test abs(B_coeffs_wf[2][1] - B_coeffs_wr[1][1]) < 1.0e-70
    @test abs(B_coeffs_wf[2][2] - B_coeffs_wr[1][2]) < 1.0e-70
    @test norm(B_HSf[1][1] - B_HSwr[2][1]) < 1.0e-70
    @test norm(B_HSf[2][1] - B_HSwr[1][1]) < 1.0e-70
    @test norm(B_HSf[1][2] - B_HSwr[2][2]) < 1.0e-70
    @test norm(B_HSf[2][2] - B_HSwr[1][2]) < 1.0e-70
    @test norm(B_spdata_f1[1] - B_spdata_wr2[1]) < 1.0e-70
    @test norm(B_spdata_f1[2] - B_spdata_wr2[2]) < 1.0e-70
    @test norm(B_spdata_f1[3] - B_spdata_wr2[3]) < 1.0e-70
    @test norm(B_spdata_f2[1] - B_spdata_wr1[1]) < 1.0e-70
    @test norm(B_spdata_f2[2] - B_spdata_wr1[2]) < 1.0e-70
    @test norm(B_spdata_f2[3] - B_spdata_wr1[3]) < 1.0e-70
    @test abs(B_coeffs_f[1][1] - B_coeffs_wr[2][1]) < 1.0e-70
    @test abs(B_coeffs_f[1][2] - B_coeffs_wr[2][2]) < 1.0e-70
    @test abs(B_coeffs_f[2][1] - B_coeffs_wr[1][1]) < 1.0e-70
    @test abs(B_coeffs_f[2][2] - B_coeffs_wr[1][2]) < 1.0e-70


    gw = SimpleWeightedGraph(6)
    add_edge!(gw,1,2,2)
    add_edge!(gw,2,3,2)
    add_edge!(gw,3,4,2)
    add_edge!(gw,4,5,2)
    add_edge!(gw,5,6,2)
    add_edge!(gw,2,4,2)
    add_edge!(gw,3,5,2)
    g = SimpleGraph(adjacency_matrix(gw))
    HS = forward_hierarchical_structure(g)
    spdata = findnz(HS[3])
    coeffs = hierarchical_coefficients(g,HS)
    @test norm(HS[1] - [-1.0, 1.0/3.0, 2.0/3.0, 2.0/3.0, 1.0/3.0, -1.0]) < 1.0e-14
    @test norm(HS[2] - [7.0/3.0, 7.0/9.0, 7.0/9.0, 7.0/9.0, 7.0/9.0, 7.0/3.0]) < 1.0e-14
    @test norm(spdata[1] - [2,1,3,4,2,4,5,2,3,5,3,4,6,5]) < 1.0e-14
    @test norm(spdata[2] - [1,2,2,2,3,3,3,4,4,4,5,5,5,6]) < 1.0e-14
    @test norm(spdata[3] -[-4.0/3.0,4.0/3.0,-1.0/3.0,-1.0/3.0,1.0/3.0,0.0,1.0/3.0,1.0/3.0,0.0,1.0/3.0,-1.0/3.0,-1.0/3.0,4.0/3.0,-4.0/3.0]) < 1.0e-14
    @test abs(coeffs[1] - 1.0) < 1.0e-14
    @test abs(coeffs[2] - 0.7559289460184544) < 1.0e-14
    HSA = forward_hierarchical_structure(adjacency_matrix(g))
    spdataA = findnz(HSA[3])
    coeffsA = hierarchical_coefficients(g,HSA)
    @test norm(HSA[1] - HS[1]) < 1.0e-14
    @test norm(HSA[2] - HS[2]) < 1.0e-14
    @test norm(spdataA[1] - spdata[1]) < 1.0e-14
    @test norm(spdataA[2] - spdata[2]) < 1.0e-14
    @test norm(spdataA[3] - spdata[3]) < 1.0e-14
    @test abs(coeffsA[1] - coeffs[1]) < 1.0e-14
    @test abs(coeffsA[2] - coeffs[2]) < 1.0e-14
    HSw = forward_hierarchical_structure(gw)
    spdataw = findnz(HSw[3])
    coeffsw = hierarchical_coefficients(gw,HSw)
    @test norm(HS[1] - HSw[1]) < 1.0e-14
    @test norm(HS[2] - HSw[2]) < 1.0e-14
    @test norm(spdata[1] - spdata[1]) < 1.0e-14
    @test norm(spdata[2] - spdata[2]) < 1.0e-14
    @test norm(spdata[3] - spdata[3]) < 1.0e-14
    @test abs(coeffs[1] - coeffs[1]) < 1.0e-14
    @test abs(coeffs[2] - coeffs[2]) < 1.0e-14


    gw = SimpleWeightedDiGraph(4)
    add_edge!(gw,1,2,1)
    add_edge!(gw,2,3,1)
    add_edge!(gw,3,4,1)
    add_edge!(gw,4,1,5)
    g = SimpleDiGraph(adjacency_matrix(gw))
    HS = forward_hierarchical_structure(g)
    spdata = findnz(HS[3])
    coeffs = hierarchical_coefficients(g,HS)
    @test norm(HS[1]) < 1.0e-14
    @test norm(HS[2] - [1,1,1,1]) < 1.0e-14
    @test norm(spdata[1] - [4,1,2,3]) < 1.0e-14
    @test norm(spdata[2] - [1,2,3,4]) < 1.0e-14
    @test norm(spdata[3]) < 1.0e-14
    @test abs(coeffs[1] - 1) < 1.0e-14
    @test abs(coeffs[2]) < 1.0e-14
    HS_B = forward_hierarchical_structure(g, big=true)
    spdata_B = findnz(HS_B[3])
    coeffs_B = hierarchical_coefficients(g,HS_B)
    @test norm(HS_B[1]) < 1.0e-75
    @test norm(HS_B[2] - [1,1,1,1]) < 1.0e-75
    @test norm(spdata_B[1] - [4,1,2,3]) < 1.0e-75
    @test norm(spdata_B[2] - [1,2,3,4]) < 1.0e-75
    @test norm(spdata_B[3]) < 1.0e-75
    @test abs(coeffs_B[1] - 1) < 1.0e-75
    @test abs(coeffs_B[2]) < 1.0e-75
    HS_w = forward_hierarchical_structure(gw)
    spdata_w = findnz(HS_w[3])
    coeffs_w = hierarchical_coefficients(g,HS_w)
    @test norm(HS_w[1] - [9/19, 3/19, -(3/19), -(9/19)]) < 1.0e-14
    @test norm(HS_w[2] - [1/19,25/19,25/19,25/19]) < 1.0e-14
    @test norm(spdata_w[1] - [4,1,2,3]) < 1.0e-14
    @test norm(spdata_w[2] - [1,2,3,4]) < 1.0e-14
    @test norm(spdata_w[3] - [18/19,-(6/19),-(6/19),-(6/19)]) < 1.0e-14
    @test abs(coeffs_w[1] - 1) < 1.0e-14
    @test abs(coeffs_w[2] - sqrt(3)*6/19) < 1.0e-14
    HS_wB = forward_hierarchical_structure(gw, big=true)
    spdata_wB = findnz(HS_wB[3])
    coeffs_wB = hierarchical_coefficients(g,HS_wB)
    @test norm(HS_wB[1] - [big(9)/19, big(3)/19, -(big(3)/19), -(big(9)/19)]) < 1.0e-75
    @test norm(HS_wB[2] - [big(1)/19,big(25)/19,big(25)/19,big(25)/19]) < 1.0e-75
    @test norm(spdata_wB[1] - [4,1,2,3]) < 1.0e-75
    @test norm(spdata_wB[2] - [1,2,3,4]) < 1.0e-75
    @test norm(spdata_wB[3] - [big(18)/19,-(big(6)/19),-(big(6)/19),-(big(6)/19)]) < 1.0e-75
    @test abs(coeffs_wB[1] - 1) < 1.0e-75
    @test abs(coeffs_wB[2] - sqrt(big(3))*6/19) < 1.0e-75
    gwr = SimpleWeightedDiGraph(transpose(weights(gw)))
    gr = SimpleDiGraph(transpose(adjacency_matrix(gw)))
    HS_F = hierarchical_structure(g)
    HS_R = hierarchical_structure(gr)
    HS_wF = hierarchical_structure(gw)
    HS_wR = hierarchical_structure(gwr)
    coeffs_F = hierarchical_coefficients(g, HS_F)
    coeffs_R = hierarchical_coefficients(gr, HS_R)
    coeffs_wF = hierarchical_coefficients(gw, HS_wF)
    coeffs_wR = hierarchical_coefficients(gwr, HS_wR)
    @test coeffs_F[1][1] - coeffs_F[2][1] < 1.0e-14
    @test coeffs_F[1][2] - coeffs_F[2][2] < 1.0e-14
    @test coeffs_F[2][1] - coeffs_F[1][1] < 1.0e-14
    @test coeffs_F[2][2] - coeffs_F[1][2] < 1.0e-14
    @test coeffs_wF[1][1] - coeffs_wF[2][1] < 1.0e-14
    @test coeffs_wF[1][2] - coeffs_wF[2][2] < 1.0e-14
    @test coeffs_wF[2][1] - coeffs_wF[1][1] < 1.0e-14
    @test coeffs_wF[2][2] - coeffs_wF[1][2] < 1.0e-14
    spdata_F1 = findnz(HS_F[1][3])
    spdata_F2 = findnz(HS_F[2][3])
    spdata_R1 = findnz(HS_R[1][3])
    spdata_R2 = findnz(HS_R[2][3])
    @test norm(HS_F[1][1] - HS_R[2][1]) < 1.0e-14
    @test norm(HS_F[2][1] - HS_R[1][1]) < 1.0e-14
    @test norm(HS_F[1][2] - HS_R[2][2]) < 1.0e-14
    @test norm(HS_F[2][2] - HS_R[1][2]) < 1.0e-14
    @test norm(spdata_F1[1] - spdata_R2[1]) < 1.0e-14
    @test norm(spdata_F1[2] - spdata_R2[2]) < 1.0e-14
    @test norm(spdata_F1[3] - spdata_R2[3]) < 1.0e-14
    @test norm(spdata_F2[1] - spdata_R1[1]) < 1.0e-14
    @test norm(spdata_F2[2] - spdata_R1[2]) < 1.0e-14
    @test norm(spdata_F2[3] - spdata_R1[3]) < 1.0e-14
    spdata_wF1 = findnz(HS_wF[1][3])
    spdata_wF2 = findnz(HS_wF[2][3])
    spdata_wR1 = findnz(HS_wR[1][3])
    spdata_wR2 = findnz(HS_wR[2][3])
    @test norm(HS_wF[1][1] - HS_wR[2][1]) < 1.0e-14
    @test norm(HS_wF[2][1] - HS_wR[1][1]) < 1.0e-14
    @test norm(HS_wF[1][2] - HS_wR[2][2]) < 1.0e-14
    @test norm(HS_wF[2][2] - HS_wR[1][2]) < 1.0e-14
    @test norm(spdata_wF1[1] - spdata_wR2[1]) < 1.0e-14
    @test norm(spdata_wF1[2] - spdata_wR2[2]) < 1.0e-14
    @test norm(spdata_wF1[3] - spdata_wR2[3]) < 1.0e-14
    @test norm(spdata_wF2[1] - spdata_wR1[1]) < 1.0e-14
    @test norm(spdata_wF2[2] - spdata_wR1[2]) < 1.0e-14
    @test norm(spdata_wF2[3] - spdata_wR1[3]) < 1.0e-14
    HS_F = hierarchical_structure(g, big=true)
    HS_R = hierarchical_structure(gr, big=true)
    HS_wF = hierarchical_structure(gw, big=true)
    HS_wR = hierarchical_structure(gwr, big=true)
    coeffs_F = hierarchical_coefficients(g, HS_F)
    coeffs_R = hierarchical_coefficients(gr, HS_R)
    coeffs_wF = hierarchical_coefficients(gw, HS_wF)
    coeffs_wR = hierarchical_coefficients(gwr, HS_wR)
    @test coeffs_F[1][1] - coeffs_F[2][1] < 1.0e-74
    @test coeffs_F[1][2] - coeffs_F[2][2] < 1.0e-74
    @test coeffs_F[2][1] - coeffs_F[1][1] < 1.0e-74
    @test coeffs_F[2][2] - coeffs_F[1][2] < 1.0e-74
    @test coeffs_wF[1][1] - coeffs_wF[2][1] < 1.0e-74
    @test coeffs_wF[1][2] - coeffs_wF[2][2] < 1.0e-74
    @test coeffs_wF[2][1] - coeffs_wF[1][1] < 1.0e-74
    @test coeffs_wF[2][2] - coeffs_wF[1][2] < 1.0e-74
    spdata_F1 = findnz(HS_F[1][3])
    spdata_F2 = findnz(HS_F[2][3])
    spdata_R1 = findnz(HS_R[1][3])
    spdata_R2 = findnz(HS_R[2][3])
    @test norm(HS_F[1][1] - HS_R[2][1]) < 1.0e-74
    @test norm(HS_F[2][1] - HS_R[1][1]) < 1.0e-74
    @test norm(HS_F[1][2] - HS_R[2][2]) < 1.0e-74
    @test norm(HS_F[2][2] - HS_R[1][2]) < 1.0e-74
    @test norm(spdata_F1[1] - spdata_R2[1]) < 1.0e-74
    @test norm(spdata_F1[2] - spdata_R2[2]) < 1.0e-74
    @test norm(spdata_F1[3] - spdata_R2[3]) < 1.0e-74
    @test norm(spdata_F2[1] - spdata_R1[1]) < 1.0e-74
    @test norm(spdata_F2[2] - spdata_R1[2]) < 1.0e-74
    @test norm(spdata_F2[3] - spdata_R1[3]) < 1.0e-74
    spdata_wF1 = findnz(HS_wF[1][3])
    spdata_wF2 = findnz(HS_wF[2][3])
    spdata_wR1 = findnz(HS_wR[1][3])
    spdata_wR2 = findnz(HS_wR[2][3])
    @test norm(HS_wF[1][1] - HS_wR[2][1]) < 1.0e-74
    @test norm(HS_wF[2][1] - HS_wR[1][1]) < 1.0e-74
    @test norm(HS_wF[1][2] - HS_wR[2][2]) < 1.0e-74
    @test norm(HS_wF[2][2] - HS_wR[1][2]) < 1.0e-74
    @test norm(spdata_wF1[1] - spdata_wR2[1]) < 1.0e-74
    @test norm(spdata_wF1[2] - spdata_wR2[2]) < 1.0e-74
    @test norm(spdata_wF1[3] - spdata_wR2[3]) < 1.0e-74
    @test norm(spdata_wF2[1] - spdata_wR1[1]) < 1.0e-74
    @test norm(spdata_wF2[2] - spdata_wR1[2]) < 1.0e-74
    @test norm(spdata_wF2[3] - spdata_wR1[3]) < 1.0e-74
end
