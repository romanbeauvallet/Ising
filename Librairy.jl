#!/usr/bin/env julia

using LinearAlgebra
using TensorOperations
using Plots
using HCubature
using SpecialFunctions

"""
d -- physical dimension
D -- bond dimension
return an randomly initialized MPS with N sites
"""
function init_random_mps(N::Int64, d::Int64, D::Int64)
    MPS = Vector{Array{Float64,3}}(undef, N)
    MPS[1] = randn(d, 1, D)
    for i in 2:N-1
        MPS[i] = randn(d, D, D)
    end
    MPS[N] = randn(d, D, 1)
    return MPS
end

"""
N -- number of sites
d -- physical dimension
D -- bond dimension
a -- dimension of the ancilla states
Return a random MPS of N sites with an ancilla state of dimension a, return also the contraction over the bond indices
"""
function ancilla(N::Int64, d::Int64, D::Int64, a::Int64)
    Mat = Vector{Array{}}(undef, N)
    Mat[1] = rand(a, d, 1, D)
    for i in 2:N-1
        Mat[i] = rand(a, d, D, D)
    end
    Mat[N] = rand(a, d, D, 1)
    return Mat
end

"""
N -- number of sites
d -- physical dimension
D -- bond dimension
a -- dimension of the ancilla states
Return a random MPS of N sites with an ancilla state of dimension a
"""
function ancillaBell(N::Int64, key::String; D::Int64=1, d::Int64=2, a::Int64=2)
    Mat = Vector{Array{}}(undef, N)
    for i in 1:N
        Mat[i] = Bell[key]
    end
    return Mat
end

"""
return the dagger of the MPS represented by the array M, be aware of the original indices indexing
"""
function dagger(M::Vector{Array{Float64,3}})
    m = length(M)
    Mbis = Vector{eltype(M)}(undef, m)
    for i in 1:m
        Mbis[i] = permutedims(conj(M[i]), (1, 3, 2)) # permute the axes to match the dagger operation
        #Mbis[i] = reshape(Mbis[i], size(M[i])...) # reshape to the original size
    end
    return Mbis
end

"""
Normalize the MPS represented by the array M
"""
function normalize(M::Vector{Array{}})
    N = length(M)
    G = dagger(M)
    C = tensorcontract(G[1], [1, -1, 2], M[1], [1, 2, -2])
    #println("size C", size(C))
    for i in 2:N-1
        x = tensorcontract(G[i], [3, -1, -2], M[i], [3, -3, -4])
        #println("size x", size(x))
        C = tensorcontract(C, [1, 2], x, [-1, 1, 2, -2])
        #println("norme partielle", norm(C))
    end
    p = tensorcontract(G[N], [1, 2, -1], M[N], [1, -2, 2])
    C = tensorcontract(C, [1, 2], p, [1, 2])
    return C[]
end

"""
return le svd tronqué
A, C -- matrices to be truncated
B -- vector of singular values
Dmax -- maximum bond dimension
cutoff -- threshold for truncation, every value under cutoff are erased
rejected_weight -- weight you want to reject in the truncation
"""
function tronquer(A, B, C, Dmax, cutoff, rejected_weight)
    #cutoff
    #@show B
    cuteff = norm(B) * cutoff
    p = maximum(findall(x -> x > cuteff, B))
    #@show p
    K = B[1:p] #S est ordonné dans l'ordre décroissant, on enlève les valeurs sous le cutoff
    n = length(K)
    #poids rejeté
    s = 0
    i = n
    while s < rejected_weight && i > 1
        s += K[i]
        i -= 1
    end
    #@show i
    q = minimum([Dmax, i + 1]) #on tronque à Dmax ou à i avec i+1 car boucle while
    K = K[1:q]
    #@show q
    return A[:, 1:q], K, C[:, 1:q] #attention on utilise C' donc on cut les colonnes de C
end

"""
mps -- a matrix product state represented as a vector of tensors
return the left canonical form of this tensor
"""
function canonicalleft!(mps) #(physique, gauche, droite)
    n = length(mps)
    mPS_canonical = Vector{eltype(mps)}(undef, n)
    leftcenter = Vector{eltype(mps)}(undef, n)
    leftcenter[1] = mps[1] # first tensor remains unchanged
    for i in 1:n-1
        (d, D, h) = size(mps[i]) # get the physical and bond dimensions
        A = reshape(mps[i], (d * D, h))
        U, S, V = svd(A)
        r = size(U, 2)
        mPS_canonical[i] = reshape(U, (d, D, r))
        q = size(mps[i+1], 3)
        mps[i+1] = tensorcontract(mps[i+1], (-1, 1, -3), Diagonal(S) * V', (-2, 1))# reshape to the original size, le reshape est obligatoire et vient de l'écriture de la contraction avec tensorcontract
        mps[i+1] = reshape(mps[i+1], (d, r, q)) # reshape to the original size
        leftcenter[i+1] = mps[i+1] # store the left center tensor
    end
    mPS_canonical[n] = mps[n] # last tensor remains unchanged
    leftcenter[n] = mps[n] # store the last tensor as the left center tensor
    return [mPS_canonical, leftcenter]
end

"""
mps -- vector of 3D tensors
return the right canonical form of this tensor
"""
function canonicalright!(mps) #(physique, gauche, droite)
    n = length(mps)
    mPS_canonical = Vector{eltype(mps)}(undef, n)#pas initialisé comme ça
    rightcenter = Vector{eltype(mps)}(undef, n)
    rightcenter[n] = mps[n] # last tensor remains unchanged
    for i in n:-1:2
        (d, D, h) = size(mps[i])
        C = permutedims(mps[i], (2, 1, 3)) # permute the axes to match the right canonical form
        A = reshape(C, (D, d * h))
        U, S, V = svd(A)
        r = size(V', 1)
        mPS_canonical[i] = permutedims(reshape(V', (r, d, h)), (2, 1, 3)) # reshape and permute to the original size
        mps[i-1] = tensorcontract(mps[i-1], (-1, -2, 1), U * Diagonal(S), (1, -3))
        rightcenter[i-1] = mps[i-1] # store the right center tensor
    end
    mPS_canonical[1] = mps[1] # first tensor remains unchanged
    rightcenter[1] = mps[1] # store the first tensor as the right center tensor
    return [mPS_canonical, rightcenter]
end

"""
vérifie si les MPS sont dans la forme canonique en contractant les deux mps
en pratique on prend mps2 = dagger mps1
les deux mps doivent être dans la même jauge et être deja canonique
"""
function contractcanon(Mps1, Mps2, side, boundary)
    n, m = length(Mps1), length(Mps2)
    if n != m
        error("The number of tensors in Mps1 and Mps2 must be the same.")
    end
    if side == "left"
        if boundary >= n
            error("boundary must be less than n")
        end
        if n > 1
            C = tensorcontract(Mps1[1], (1, 2, -1), Mps2[1], (1, -2, 2))
            for i in 2:boundary
                x = tensorcontract(Mps1[i], (3, -1, -2), Mps2[i], (3, -3, -4))
                C = tensorcontract(C, (1, 2), x, (1, -1, -2, 2))
            end
            return C, size(C)
        end
    elseif side == "right"
        if boundary <= 1
            error("boundary must be greater than 1")
        end
        if n > 1
            C = tensorcontract(Mps1[n], (1, -1, 2), Mps2[n], (1, 2, -2))
            for i in n-1:-1:boundary
                x = tensorcontract(Mps1[i], (1, -1, -2), Mps2[i], (1, -3, -4))
                C = tensorcontract(C, (1, 2), x, (-1, 1, 2, -2))
            end
            return C, size(C)
        end
    else
        error("side must be 'left' or 'right'")
    end
end

"""
return the new mps after a sweep of tebd on tow gates
tensor type = (physical, left, right)
gate type = (up left, up right, down left, down right)
"""
function tebd_step(left, right, gate, side::String, Dmax, cutoff, rejected_weight)
    Dleft = size(left, 2)
    Dright = size(right, 3)
    dleft = size(left, 1)
    dright = size(right, 1)

    if dleft != size(gate, 1) || dright != size(gate, 2)
        error("The dimensions of the tensors do not match.")
    elseif size(left, 3) != size(right, 2)
        error("The bond dimensions of the tensors do not match.")
    end
    @assert dleft == size(gate, 3)
    @assert dright == size(gate, 4)

    mid = tensorcontract((-1, -2, -3, -4), left, (-1, -2, 1), right, (-3, 1, -4)) #(physique, gauche, physique, droite)
    t = tensorcontract((-3, -1, -4, -2), mid, (1, -1, 2, -2), gate, (1, 2, -3, -4)) #T type = (gauche haut, droite haut, bas gauche, bas droite)
    @assert size(t) == (dleft, Dleft, dright, Dright)
    m = reshape(t, dleft * Dleft, dright * Dright)
    R, Y, Q = svd(m)
    U, S, V = tronquer(R, Y, Q, Dmax, cutoff, rejected_weight) #troncation step
    newDmid = length(S)
    normalize!(S) #normalisation step
    #on fait la svd pour le nouveau mps
    if side == "right"
        A0 = U
        B0 = Diagonal(S) * V'
    elseif side == "left"
        A0 = U * Diagonal(S)
        B0 = V'
    else
        error("side must be 'left' or 'right'")
    end
    A = reshape(A0, dleft, Dleft, newDmid) #(physical, left, bond)
    B = reshape(B0, newDmid, dright, Dright)
    B = permutedims(B, (2, 1, 3))
    return A, B
end

"""
beta -- inverse temperature
compute and return the gibbs function of the 2D infinite Ising model
"""
function exactenergylibre(beta)
    f(x) = -(log(2) + log(cosh(2 * beta * J)^2) - sinh(2 * beta * J) * (cos(x[1]) + cos(x[2])) / (8 * pi)) / beta
    val, err = hcubature(f, [0, 0], [2 * pi, 2 * pi])
    return val, err
end

"""
beta -- inverse temperature
compute and return the exact energy of the 2D infinite Ising model
"""
function exactenergy(beta)
    k = 1 / (sinh(2 * beta)^2)
    k_modulus = 4 * k / (1 + k)^2
    K = ellipk(k_modulus)
    return -(1 + 2 / pi * (2 * tanh(2 * beta)^2 - 1) * K) / tanh(2 * beta)
end

"""
beta -- inverse temperature
J -- coupling constant
h -- disorder
return the Boltzmann matrix (matrix with Boltzmann weight + disorder)
"""
function isingMatrix(beta, J, h=0) #signe doit etre cohérent avec la convention du tenseur de l'énergie
    # Create the Ising matrix for a single site
    M = exp(beta * J) * LinearAlgebra.I(2) # Identity matrix
    return M + exp(-beta * J) * [0 1; 1 0] + h * [1 0; 0 -1]
end

"""
j -- coupling constant
return the energy operator
"""
function energytensor(j)
    return [-j j; j -j]
end

"""
beta -- inverse temperature
j -- coupling constant
h -- disorder
return the Ising tensor for the evolution of the mps
"""
function isingTensor(beta, j, h)
    D = zeros(Int, 2, 2, 2, 2)
    for i in 1:2
        D[i, i, i, i] = 1
    end
    M = sqrt(isingMatrix(beta, j, h))
    @tensor T[i, j, k, l] := D[a, b, c, d] * M[i, a] * M[j, b] * M[k, c] * M[l, d]
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