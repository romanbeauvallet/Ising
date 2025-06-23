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

J = 1.0
h = 0.0
beta = 0.1
n = 10
N = 20 #ATTENTION N doit être pair
d = 2
D0 = 4
Dmax = 100
cutoff = 1e-12
rejected_weight = 1e-20

include("Codes.jl")
# Ising model in 2D with open boundary conditions

function isingMatrix(beta, J, h=0)
    # Create the Ising matrix for a single site
    M = exp(beta*J)*LinearAlgebra.I(2) # Identity matrix
    return M + exp(-beta*J)*[0 1; 1 0] + h*[1 0; 0 -1]
end

function energytensor(j)
    return [j -j ; -j j]
end
function isingTensor(beta, j, h)
    D = zeros(Int, 2, 2, 2, 2)
    for i in 1:2
        D[i, i, i, i] = 1
    end
    M = sqrt(isingMatrix(beta, j, h))
    @tensor T[i,j,k,l] := D[a, b, c, d] * M[i,a] * M[j,b] * M[k,c] * M[l,d]
    return T
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
function ising2D(N, D0, d, J, h, β, n_sweeps, Dmax, rejected_weight, cutoff)
    #@show "Begin"
    gate = isingTensor(β, J, h) # Example parameters for the Ising tensor
    mps = init_random_mps(N, d, D0)
    for i in 1:n_sweeps
        #@show "SWEEP i =", i
        mps = tebd_sweep(mps, gate, Dmax, rejected_weight, cutoff)
    end
    return mps
end

"""
MPS - mettre le MPS final avec l'évolution avec le tenseur d'Ising il est censé être en right canonical donc avec toute l'information dans le tenseur de gauche au bout
beta, J, h - paramètre physiques
i - site du MPS sur lequel on calcule l'énergie

return l'énergie sur le site i du MPS
"""
function energyIsing(mps, J, i_meas)#mettre un ! au debut du nom de la fonction pour spécifier que la fonction modifie l'input
    #mettre en left et right canonical par rapport au tenseur i
    #garder seulement le centre car toute l'information du mps y est
    #contracter entre le tenseur et le dagger la matrice énergie et contracter tous les axes
    #vérifier que la norme du mps est bien 1 (si pas le cas normaliser la contraction précédente par la norme du tenseur)

    #%% orthonormalisation
    N = length(mps)
    if !(1 < i_meas < N)
        throw(ArgumentError("site is not in the mps"))
    end
    bond_operator = energytensor(J)
    #@show size(MPS[1:i]), size(canonicalleft(MPS[1:i])[1])
    center_canonical_mps = deepcopy(mps)
    center_canonical_mps[begin:i_meas], _ = canonicalleft!(center_canonical_mps[begin:i_meas])
    #verif
    I = tensorcontract(center_canonical_mps[i_meas-1], (1, 2, -1), conj(center_canonical_mps[i_meas-1]), (1, 2, -2))
    @assert I ≈ LinearAlgebra.I(size(I, 1))
    I2 = tensorcontract(center_canonical_mps[i_meas+1], (1, -1, 2), conj(center_canonical_mps[i_meas+1]), (1, -2, 2))
    @assert I2 ≈ LinearAlgebra.I(size(I2, 1))
    #%% contraction
    ket = center_canonical_mps[i_meas]
    #@show size(ket), typeof(ket)
    #mps_norm = tensorcontract(ket, (1, 2, 3), conj(ket), (1, 2, 3))
    #@show mps_norm
    inter = tensorcontract((-1, -2, -3), conj(ket), (1, -2, -3), bond_operator, (1, -1))
    E = tensorcontract(inter, (1, 2, 3), ket, (1, 2, 3))[]
    return E
end



site_measure = N ÷ 2
Betalist = collect(0.1:0.1:2)
#Betalist = [0.1, 0.2]
Eexact = ExactEnergy.(Betalist)

MPSlist = map(β -> ising2D(N, D0, d, J, h, β, 100, Dmax, rejected_weight, cutoff), Betalist);
Elist = map(mps -> energyIsing(mps, J, site_measure), MPSlist)

#on fait les données et on trace
gr()
plot(Betalist, -2*Elist, label="TEBD", xlabel="\$\\beta\$", ylabel="Energy")
plot!(Betalist, Eexact, label="exact")
