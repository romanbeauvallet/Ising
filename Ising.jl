#!/usr/bin/env julia
using LinearAlgebra
using TensorOperations
using Plots
using HCubature
using SpecialFunctions

function ExactEnergylibre(b)
    f(x) = -(log(2) + log(cosh(2 * b * J)^2) - sinh(2 * b * J) * (cos(x[1]) + cos(x[2])) / (8 * pi)) / b
    val, err = hcubature(f, [0, 0], [2 * pi, 2 * pi])
    return val, err
end

function ExactEnergy(beta)
    k = 1 / (sinh(2 * beta)^2)
    k_modulus = 4 * k / (1 + k)^2
    K = ellipk(k_modulus)
    return -(1 + 2 / pi * (2 * tanh(2 * beta)^2 - 1) * K) / tanh(2 * beta)
end

function exactmagnetization(beta, j=J)
    betacritic = log(1 + sqrt(2)) / (2 * j)
    if beta < betacritic
        return 0
    else
        return (1 - (sinh(2 * j * beta))^(-4))^(1 / 8)
    end
end

include("Codes.jl")
# Ising model in 2D with open boundary conditions

function isingMatrix(beta, J, h=0)
    # Create the Ising matrix for a single site
    M = exp(beta * J) * LinearAlgebra.I(2) # Identity matrix
    return M + exp(-beta * J) * [0 1; 1 0] + h * [1 0; 0 -1]
end

function isingMatrix2(beta, j, h=0)
    M = [cosh(2 * beta * j) sinh(2 * beta * j); sinh(2 * beta * j) cosh(2 * beta * j)]
    return exp(-2 * beta * j) * M
end
function energytensor(j)
    return j * [1 -1; -1 1]
end
function isingTensor(beta, j, h)
    D = zeros(Int, 2, 2, 2, 2)
    for i in 1:2
        D[i, i, i, i] = 1
    end
    M = sqrt(isingMatrix(beta, j, h))
    @tensor T[i, j, k, l] := D[a, b, c, d] * M[i, a] * M[j, b] * M[k, c] * M[l, d]
    return T
end

function isingTensor2(beta, j, h)
    D = zeros(Int, 2, 2, 2, 2)
    for i in 1:2
        D[i, i, i, i] = 1
    end
    M = sqrt(isingMatrix2(beta, j, h))
    @tensor T[i, j, k, l] := D[a, b, c, d] * M[i, a] * M[j, b] * M[k, c] * M[l, d]
    return T
end

function tensormagnetize(beta, j, h)
    D = zeros(Int, 2, 2, 2, 2)
    D[1, 1, 1, 1] = 1
    D[2, 2, 2, 2] = -1
    M = sqrt(isingMatrix(beta, j, h))
    @tensor T[i, j, k, l] := D[a, b, c, d] * M[i, a] * M[j, b] * M[k, c] * M[l, d]
    return T
    @show size(T)
end

"""
one sweep of TEBD for the 2D Ising model with a different method
always start with a right canonical mps, sweep : left->right->left

"""
function tebd_sweep(mps, gate, Dmax, rejected_weight, cutoff)
    #@show "BEGIN LEFT TO RIGHT SWEEP"
    N = length(mps)
    #%% FIRST PART ON EVEN COUPLE
    mpsnew = Vector{eltype(mps)}(undef, N)
    right = canonicalright!(mps)[1]
    #println.(size.(right))
    for i in 1:2:N-3 #n est even
        #@show i, i + 1
        #@show i, size(right[i])
        mpsnew[i], temp = tebd_step(right[i], right[i+1], gate, "right", Dmax, cutoff, rejected_weight)
        #@show size(mpsnew[i])
        #changement de jauge
        U, S, V = svd(reshape(temp, size(temp, 1) * size(temp, 2), size(temp, 3)))
        mpsnew[i+1] = reshape(U, size(temp, 1), size(temp, 2), size(U, 2))
        right[i+2] = tensorcontract((-2, -1, -3), Diagonal(S) * V', (-1, 1), right[i+2], (-2, 1, -3))
    end
    #@show "special case", N - 1, N
    #il reste les deux derniers à faire à la main ou alors on rallonge le mps de 1 tenseur au départ qui rammasse les poids de la svd puis on cut
    mpsnew[N-1], last_site = tebd_step(right[N-1], right[N], gate, "right", Dmax, cutoff, rejected_weight)
    #mettre le dernier dans la jauge gauche
    last_site = permutedims(last_site, (2, 3, 1))  # left-right-phys
    A, B, C = svd(reshape(last_site, size(last_site, 1), size(last_site, 2) * size(last_site, 3)))
    mpsnew[N] = permutedims(reshape(C', size(C', 1), size(last_site, 2), size(last_site, 3)), (3, 1, 2))
    mpsnew[N-1] = tensorcontract((-1, -2, -3), mpsnew[N-1], (-1, -2, 1), A * Diagonal(B), (1, -3))
    #@show "BEGIN RIGHT TO LEFT SWEEP"
    for i in N-1:-2:3
        #@show i - 1, i
        #@show i
        #@show size(right[i-1]), size(right[i])
        temp, mpsnew[i] = tebd_step(mpsnew[i-1], mpsnew[i], gate, "left", Dmax, cutoff, rejected_weight)
        #changement de jauge
        temp = permutedims(temp, (2, 1, 3)) # Permute dimensions to reshape correctly (gauche, physique, droite)
        U, S, V = svd(reshape(temp, size(temp, 1), size(temp, 2) * size(temp, 3)))
        mpsnew[i-1] = permutedims(reshape(V', size(V', 1), size(temp, 2), size(temp, 3)), (2, 1, 3))
        mpsnew[i-2] = tensorcontract((-1, -2, -3), mpsnew[i-2], (-1, -2, 1), U * Diagonal(S), (1, -3))
    end
    return mpsnew
end

"""
N - number of sites
d - dimension of physical indices
D - bond dimension
J - coupling constant
h - external magnetic field
beta - inverse tempsuperature
n - number of sweeps
cutoff - convergence criterion
Dmax - maximum bond dimension
side - side of the lattice

returns the final MPS for the 2D Ising model after applying TEBD and n sweeps
"""
function ising2D(mpsin, J, h, β, n_sweeps, Dmax, rejected_weight, cutoff)
    #@show "Begin"
    gate = isingTensor(β, J, h) # Example parameters for the Ising tensor
    mps = deepcopy(mpsin)
    for i in 1:n_sweeps
        #@show i
        mps = tebd_sweep(mps, gate, Dmax, rejected_weight, cutoff)
    end
    return mps
end

"""
MPS - mettre le MPS final avec l'évolution avec le tenseur d'Ising il est censé être en right canonical donc avec toute l'information dans le tenseur de gauche au bout
beta, J, h - paramètre physiques
i - site du MPS sur lequel on calcule l'énergie
gate -- magnetization tensor
gatenorm -- ising tensor to compute norm of the environment

on prend la gate dans l'ordre (gauche, up, droite, down)

return l'énergie sur le site i du MPS
"""
function energyIsing!(mps, J, i_meas, gate, gatenorm)#mettre un ! au debut du nom de la fonction pour spécifier que la fonction modifie l'input
    #mettre en left et right canonical par rapport au tenseur i
    #garder seulement le centre car toute l'information du mps y est
    #contracter entre le tenseur et le dagger la matrice énergie et contracter tous les axes
    #vérifier que la norme du mps est bien 1 (si pas le cas normaliser la contraction précédente par la norme du tenseur)
    #on prend la gate dans l'ordre (gauche, up, droite, down)
    #%% orthonormalisation
    N = length(mps)
    if !(1 < i_meas < N - 2)
        throw(ArgumentError("site is not in the mps"))
    end
    mps[begin:i_meas], _ = canonicalleft!(mps[begin:i_meas])
    env_up = mps[i_meas: i_meas+2]
    env_down = dagger(env_up)
    @tensor ket[gauche, physique1, physique2, physique3, droite] := env_up[1][physique1, gauche, i] * env_up[2][physique2, i, j] * env_up[3][physique3, j, droite]
    @tensor bras[gaucheaux, physiqueaux1, physiqueaux2, physiqueaux3, droiteaux] := env_down[1][physiqueaux1, i, gaucheaux] * env_down[2][physiqueaux2, j, i] * env_down[3][physiqueaux3, droiteaux, j]
    @tensor energynonorm[] := ket[endleft, physique1, physique2, physique3, endright] * gate[physique1, physique2, i, physiqueaux1] * gate[i, physique3, physiqueaux3, physiqueaux2]*bras[endleft, physiqueaux1, physiqueaux2, physiqueaux3, endright]
    @tensor norm[] := ket[endleft, physique1, physique2, physique3, endright] * gatenorm[physique1, physique2, i, physiqueaux1] * gatenorm[i, physique3, physiqueaux3, physiqueaux2]*bras[endleft, physiqueaux1, physiqueaux2, physiqueaux3, endright]
    return only(energynonorm)/only(norm)
end

function magnetizationIsing(mps, gate, i_meas)
    n = length(mps)
    if !(1 < i_meas < n - 1)
        throw(ArgumentError("site is not in the mps"))
    end
    #@show size(MPS[1:i]), size(canonicalleft(MPS[1:i])[1])
    center_canonical_mps = deepcopy(mps)
    center_canonical_mps[begin:i_meas], _ = canonicalleft!(center_canonical_mps[begin:i_meas])
    #%% contraction
    #@show size(center_canonical_mps[i_meas])
    ket = tensorcontract((-2, -1, -3, -4), center_canonical_mps[i_meas], (-1, -2, 1), center_canonical_mps[i_meas+1], (-3, 1, -4)) #(gauche, physique, physique, droite)
    inter0 = tensorcontract((-1, -3, -4, -2), ket, (-1, 1, 2, -2), gate, (1, 2, -3, -4)) #(gauche, physique, physique, droite)
    bras = tensorcontract((-2, -1, -3, -4), conj(center_canonical_mps[i_meas]), (-1, -2, 1), conj(center_canonical_mps[i_meas+1]), (-3, 1, -4)) #meme format
    E = tensorcontract(inter0, (1, 2, 3, 4), bras, (1, 2, 3, 4))[]
    return E
end

J = 1.0
h = 0.0
n = 10
N = 30 #ATTENTION N doit être pair
d = 2
D0 = 10
Dmax = 100
cutoff = 1e-20
rejected_weight = 1e-20


site_measure = N ÷ 2
Betalist = collect(0.1:0.01:2)
#Betalist = [0.1, 0.2]
Eexact = ExactEnergy.(Betalist)
Mexact = exactmagnetization.(Betalist)

Elist = Vector{Float64}()
Mlist = Vector{Float64}()
MPSlist = Vector{}()
test = init_random_mps(N, d, D0)
#mpstestising = ising2D(test, J, h, 0.1, 100, Dmax, rejected_weight, cutoff)
#@show size.(mpstestising)
push!(MPSlist, test)
function loop()
    for i in eachindex(Betalist)
        @show Betalist[i]
        operator = tensormagnetize(Betalist[i], J, h)
        norm_operator = isingTensor(Betalist[i], J, h)
        mpsbeta = ising2D(MPSlist[i], J, h, Betalist[i], 100, Dmax, rejected_weight, cutoff)
        m = magnetizationIsing(mpsbeta, operator, site_measure)
        @show m
        e = energyIsing!(mpsbeta, J, site_measure, operator, norm_operator)
        @show e
        push!(MPSlist, mpsbeta)
        push!(Mlist, m)
        push!(Elist, e)
    end
end

loop()

#on fait les données et on trace
gr()


p1 = plot(Betalist, -2*Elist, label="TEBD", xlabel="\$\\beta\$", ylabel="Energy")
plot!(Betalist, Eexact, label="exact")
display(p1)


#p2 = plot(Betalist, abs.(Mlist), label="TEBD", xlabel="\$\\beta\$", ylabel="Magnetization")
#plot!(Betalist, Mexact, label="exact")
#display(p2)
