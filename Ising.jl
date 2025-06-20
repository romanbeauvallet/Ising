#!/usr/bin/env julia
using LinearAlgebra
using TensorOperations
using Plots
using HCubature

function ExactEnergy(b)
    f(x) = -(log(2) + log(cosh(2*b*J)^2) - sinh(2*b*J)*(cos(x[1]) + cos(x[2]))/(8*pi))/b
    val, err = hcubature(f, [0, 0], [2*pi, 2*pi])
    return val, err
end

J = 1.0
h = 0.0
Beta = 0.1
n = 10
N = 20 #ATTENTION N doit être pair
d = 2
D = 20
i = 10
dmax = 1000
cutoff = 1e-50
side = "right"
sum = 1e-50

include("Codes.jl")
# Ising model in 2D with open boundary conditions

function isingMatrix(Beta, J, h=0)
    # Create the Ising matrix for a single site
    M = exp(Beta*J)*I # Identity matrix
    return M + exp(-Beta*J)*[0 1; 1 0] + h*[1 0; 0 -1]
end

function energytensor(J)
    return [J -J ; -J J]
end 
function isingTensor(Beta, J, h)
    D = zeros(Int, 2, 2, 2, 2)
    for i in 1:2
        D[i, i, i, i] = 1
    end
    M = sqrt(isingMatrix(Beta, J, h))
    @tensor T[i,j,k,l] := D[a, b, c, d] * M[i,a] * M[j,b] * M[k,c] * M[l,d]
    return T
end

"""
N - number of sites
d - dimension of physical indices
D - bond dimension
J - coupling constant
h - external magnetic field
Beta - inverse temperature
n - number of sweeps
cutoff - convergence criterion
dmax - maximum bond dimension
side - side of the lattice
Returns the MPS for the 2D Ising model after applying TEBD for 1 sweep
"""
function ising2Dunit(MPS, J::Float64, h::Float64, Beta::Float64, n, cutoff, dmax, side, sum)
    G = isingTensor(Beta, J, h)
    k = length(MPS)
    h = deepcopy(MPS) #deepcopy to avoid modifying the original MPS
    #tout en left
    mpsleft, leftcenter= Vector{eltype(MPS)}(undef, k), Vector{eltype(MPS)}(undef, k)
    mpsleft, leftcenter = canonicalleft(MPS)
    #tout en right
    mpsright, rightcenter = Vector{eltype(MPS)}(undef, k), Vector{eltype(MPS)}(undef, k)
    mpsright, rightcenter = canonicalright(h)
    #new MPS
    mpsnew = Vector{eltype(MPS)}()
    for i in 1:2:n
        println("verif size tensors", i, " ", size(leftcenter[i]), " ", size(rightcenter[i+1]))
        println("porte", size(G))
        step = tebd(leftcenter[i], rightcenter[i+1], G, dmax, cutoff, sum, side)
        println("debug", i, " ", step)
        push!(mpsnew, step) 
    end
    #canonise encore    
    mpsleft, leftcenter = canonicalleft(mpsnew)
    mpsright, rightcenter = canonicalright(mpsnew)
    for i in 2:2:n
        mpsnew[i], mpsnew[i+1] = tebd(mpsleft[i], mpsright[i+1], G, dmax, cutoff, sum, side)
    end
end

"""
one sweep of TEBD for the 2D Ising model with a different method
always start with a right canonical mps, sweep : left->right->left

"""
function oneIsing(mps, Beta, J, h, Dmax, sum, cutoff)
    p = length(mps)
    gate = isingTensor(Beta, J, h) # Example parameters for the Ising tensor
    #%% FIRST PART ON EVEN COUPLE
    mpsnew = Vector{eltype(mps)}(undef, p)
    tempmps = Vector{eltype(mps)}(undef, p)
        right = canonicalright(mps)[1]
        #println.(size.(right))
        for i in 1:2:p-3 #n est even
            #@show i, size(right[i])
            mpsnew[i], tempmps[i+1] = tebd2(right[i], right[i+1],gate, "right", Dmax, cutoff, sum)
            #@show size(mpsnew[i])
            #changement de jauge
            U, S, V = svd(reshape(tempmps[i+1], size(tempmps[i+1], 1)*size(tempmps[i+1], 2), size(tempmps[i+1], 3)))
            mpsnew[i+1] = reshape(U, size(tempmps[i+1], 1), size(tempmps[i+1], 2), size(U, 2))
            right[i+2] = tensorcontract(Diagonal(S)*V', (-1, 1), right[i+2],(-2,1, -3))
            right[i+2] = permutedims(right[i+2], (2, 1, 3)) # Permute dimensions to match the original order
        end
        #il reste les deux derniers à faire à la main ou alors on rallonge le mps de 1 tenseur au départ qui rammasse les poids de la svd puis on cut 
        mpsnew[p-1], mpsnew[p] = tebd2(right[p-1], right[p], gate, side, Dmax, cutoff, sum)
        #mettre le dernier dans la jauge gauche
        Z = permutedims(mpsnew[p], (2,3,1))
        A, B, C = svd(reshape(Z, size(mpsnew[p], 2), size(mpsnew[p], 1)*size(mpsnew[p], 3)))
        mpsnew[p] = permutedims(reshape(C', size(C', 1), size(Z, 2), size(Z, 3)), (3, 1, 2))
        mpsnew[p-1] = tensorcontract(mpsnew[p-1], (-1, -2, 1), A*Diagonal(B), (1, -3))
        for i in p-1:-2:3
            #@show i
            #@show size(right[i-1]), size(right[i])
            tempmps[i-1], mpsnew[i] = tebd2(mpsnew[i-1], mpsnew[i], gate, "left", Dmax, cutoff, sum)
            #changement de jauge
            tempmps[i-1] = permutedims(tempmps[i-1], (2, 1, 3)) # Permute dimensions to reshape correctly (gauche, physique, droite)
            U, S, V = svd(reshape(tempmps[i-1], size(tempmps[i-1], 1), size(tempmps[i-1], 2)*size(tempmps[i-1], 3)))
            mpsnew[i-1] = permutedims(reshape(V', size(V',1), size(tempmps[i-1], 2), size(tempmps[i-1], 3)), (2,1,3))
            mpsnew[i-2] = tensorcontract(mpsnew[i-2],(-1,-2, 1),U*Diagonal(S), (1, -3))
        end
        return mpsnew
end

"""
N - number of sites
d - dimension of physical indices
D - bond dimension
J - coupling constant
h - external magnetic field
Beta - inverse temperature
n - number of sweeps
cutoff - convergence criterion
dmax - maximum bond dimension
side - side of the lattice 

returns the final MPS for the 2D Ising model after applying TEBD and n sweeps
"""
function ising2D(N::Int, d, D, J::Float64, h::Float64, Beta::Float64, n, cutoff, dmax, side, sum)
    # Create the MPS for the 2D Ising model
    mps = MPS(N, d, D)
    # Sweeps
    for i in 1:n
        mps = ising2Dunit(mps, J, h, Beta, n, cutoff, dmax, side, sum)
    end
    return mps
end

function ising2D2(N,D,d,J,h,Beta,n, Dmax, sum, cutoff)
    tensors = MPS(N,d,D)
    for i in 1:n 
        tensors = oneIsing(tensors, Beta, J, h, Dmax, sum, cutoff)
    end
    return tensors
end

"""
MPS - mettre le MPS final avec l'évolution avec le tenseur d'Ising il est censé être en right canonical donc avec toute l'information dans le tenseur de gauche au bout
Beta, J, h - paramètre physiques
i - site du MPS sur lequel on calcule l'énergie

return l'énergie sur le site i du MPS
"""
function energyIsing(mps, J, i)#mettre un ! au debut du nom de la fonction pour spécifier que la fonction modifie l'input
    #mettre en left et right canonical par rapport au tenseur i 
    #garder seulement le centre car toute l'information du mps y est 
    #contracter entre le tenseur et le dagger la matrice énergie et contracter tous les axes 
    #vérifier que la norme du mps est bien 1 (si pas le cas normaliser la contraction précédente par la norme du tenseur)

    #%% orthonormalisation
    m = length(mps)
    if !(1 < i < m)
        error("site is not in the mps")
    end
    Z = energytensor(J)
    #@show size(MPS[1:i]), size(canonicalleft(MPS[1:i])[1])
    mps[begin:i] = canonicalleft(mps[begin:i])[1]
    #verif
    I = tensorcontract(mps[i-1], (1, 2, -1), conj(mps[i-1]), (1, 2, -2))
    @show I ≈ LinearAlgebra.I(size(I, 1))
    I2 = tensorcontract(mps[i+1], (1, -1, 2), conj(mps[i+1]), (1, -2, 2))
    @show I2 ≈ LinearAlgebra.I(size(I2, 1))
    #%% contraction
    @show size(mps[i]), typeof(mps[i])
    D = permutedims(conj(mps[i]), (1,3,2))
    N = tensorcontract(mps[i], (1, 2, 3), D, (1, 3, 2))
    @show N
    inter = tensorcontract(mps[i], (1, -1, -2), Z, (1, -3))
    E = tensorcontract(inter, (1, 2, 3), D, (3, 2, 1))
    return E[], mps
end

#on fait les données et on trace
gr()

Betalist = [i for i in 0.1:0.1:10]
MPSlist = [ising2D2(N, D ,d, J ,h, Betalist[i], n, dmax, sum, cutoff) for i in eachindex(Betalist)]
Elist = [energyIsing(MPSlist[j], J, i) for j in eachindex(MPSlist)]
Eexact = [ExactEnergy(Betalist[i])[1] for i in eachindex(Betalist)]

#plot(Betalist, Elist, label="E = f(\$\\beta\$)", xlabel="\$\\beta\$", ylabel="E")
#plot!(Betalist, Eexact, label="Energie libre exacte Ising model 2D", xlabel="\$\\beta\$", ylabel="Energie")

Mps = MPS(N,d, D)
e1 = energyIsing(deepcopy(Mps), J, i)
e2 = energyIsing(deepcopy(Mps), J, i)

@show e1, e2