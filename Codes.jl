#!/usr/bin/env julia

##Librairies
using LinearAlgebra
using ITensors
#using TensorCast
using TensorOperations

##Constants

const TOL = 1e-10
sx = [0 1; 1 0] # Pauli X matrix
sy = [0 -im; im 0] # Pauli Y matrix
sz = [1 0; 0 -1] # Pauli Z matrix



Bell = Dict{String,Matrix{Float64}}()
n = sqrt(2) # normalization factor
Bell["00"] = [1/n 0; 0 1/n] # |00>
Bell["01"] = [1/n 0; 0 -1/n] # |01>
Bell["10"] = [0 1/n; 0 1/n] # |10>
Bell["11"] = [0 -1/n; 0 1/n] # |11>

##Functions

function MPSv(N::Int64, d::Int64, D::Int64)
    Mat = Dict{String,Array{Float64,3}}()
    Mat["1"] = rand(d, 1, D)
    for i in 2:N-1
        Mat["$i"] = rand(d, D, D)
    end
    Mat["$N"] = rand(d, 1, D)
    return Mat
end

#dictionary version of MPS leads to a stack overflow error

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
L -- vector of fixed physical indices (length N, values in 1:d)
Return a random MPS of N sites with fixed physical indices, return also the contraction over the bond indices
"""
function fixedMPS(N::Int64, d::Int64, D::Int64, L::Vector{Int64})
    if length(L) != d
        error("Length of L must be equal to d")
    end
    Mat = Dict{String,Matrix}()
    Mat["1"] = randn(1, d, D)
    for i in 2:N-1
        Mat["$i"] = randn(D, d, D)
    end
    Mat["$N"] = randn(D, d, 1)
    psi = Mat["1"][:, L[1], :]
    for i in 2:N-1
        psi = psi * Mat["$i"][:, L[i], :]
    end
    psi = psi * Mat["$N"][:, L[N], :]
    return psi, Mat
end

"""
N -- number of sites
d -- physical dimension
D -- bond dimension
a -- dimension of the ancilla states
Return a random MPS of N sites with an ancilla state of dimension a, return also the contraction over the bond indices
"""
function AncillaDic(N::Int64, d::Int64, D::Int64, a::Int64)
    Mat = Dict{String,Array{Float64,4}}()
    Mat["1"] = rand(a, d, 1, D)
    for i in 2:N-1
        Mat["$i"] = rand(a, d, D, D)
    end
    Mat["$N"] = rand(a, d, 1, D)
    return Mat
end

function Ancilla(N::Int64, d::Int64, D::Int64, a::Int64)
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
function AncillaBell(N::Int64, key::String; D::Int64=1, d::Int64=2, a::Int64=2)
    Mat = Vector{Array{}}(undef, N)
    for i in 1:N
        Mat[i] = Bell[key]
    end
    return Mat
end

function normalizeDic(M::Dict{String,Array{Float64,3}})
    """
    Normalize the MPS represented by the dictionary M
    """
    N = length(M)
    C = contract(conj(M["1"]), (1, 2, -1), M["1"], (1, 2, -2))
    for i in 2:N-1
        C = contract(C, (1, 2), conj(M["$i"]), (3, 2, -1), M["$i"], (3, 1, -2))
    end
    C = contract(C, (1, 2), conj(M["$N"]), (3, 2, 4), M["$N"], (3, 1, 4))
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
    #@show typeof(Mbis[i])
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
M1, M2 -- dictionaries of matrices representing MPS (no Bell states othewise its pointless)
L -- list of physical indices


each paired of matrices in M1 and M2 must have the same physical indice dimension
Return the contraction of the MPS
"""
function contraction(M1::Dict{String,Matrix}, M2::Dict{String,Matrix}, L::Vector{Int64})
    m, n = length(M1), length(M2)
    if m != n
        error("The number of matrices in M1 and M2 must be the same.")
    end
    C = localcontract(M1["1"], M2["1"])
    c = reshape(C, size(C, 1)^2, size(C, 2)^2, size(C, 3)^2)
    for i in 2:m
        key = string(i)
        @tensor R[i, j, k, l, m, n] = M1[key][i, v, j, k] * M2[key][l, v, m, n]
        r = reshape(R, size(R, 1)^2, size(R, 2)^2, size(R, 3)^2)
        @tensor c[a, b, c] = c[] * r[i, j, k, l, m, n]
    end
    return result
end
#pour la norme il suffit de prendre M2 = congugate(M1)

"""
return the Trotter-Suzuki decomposition of the matrix M with time step dt

"""
function TrotterSuzukiSpinchain(dt::Float64, N::Int, J::Complex=1.0, h=1.0)
    localhamiltonian = J * (kron(sx, sx) + kron(sy, sy) + kron(sz, sz))
    Me = Vector{Array{}}(undef, N // 2)
    Mo = Vector{Array{}}(undef, N // 2)
    G = exp(-dt * localhamiltonian)
    g = reshape(G, (2, 2, 2, 2))
    for i in 2:2:N
        push!(Me, g)
    end
    for i in 1:2:N
        push!(Mo, g)
    end
    return Mo, Me
end
"""
trotter suzuki for ising model
"""
function TrotterSuzukiIsing(dt::Float64, N::Int, J::Complex=1.0, h=0.0)
    localhamiltonian = J * (kron(sz, sz)) - h * kron(sx, I) # Ising model Hamiltonian
    Me = Vector{Array{}}()
    Mo = Vector{Array{}}()
    G = exp(-dt * localhamiltonian)
    g = reshape(G, (2, 2, 2, 2))
    for i in 2:2:N
        push!(Me, g)
    end
    for i in 1:2:N
        push!(Mo, g)
    end
    return Mo, Me
end
"""
return le svd tronqué
A, C -- matrices to be truncated
B -- vector of singular values
dmax is the maximum bond dimension
cutoff is the threshold for truncation, every value under cutoff are erased

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
MPS -- a matrix product state represented as a vector of tensors
return the left canonical form of this tensor
"""
function canonicalleft!(mps) #(physique, gauche, droite)
    n = length(mps)
    mPS_canonical = Vector{eltype(mps)}(undef, n)
    leftcenter = Vector{eltype(mps)}(undef, n)
    leftcenter[1] = mps[1] # first tensor remains unchanged
    for i in 1:n-1
        (d, D, h) = size(mps[i]) # get the physical and bond dimensions
        A = reshape(mps[i], (d*D, h))
        U, S, V = svd(A)
        r = size(U, 2)
        mPS_canonical[i] = reshape(U, (d, D, r))
        q = size(mps[i+1], 3)
        mps[i+1] =  tensorcontract(mps[i+1], (-1, 1, -3), Diagonal(S) * V', (-2,1))# reshape to the original size, le reshape est obligatoire et vient de l'écriture de la contraction avec tensorcontract
        mps[i+1] = reshape(mps[i+1], (d, r, q)) # reshape to the original size
        leftcenter[i+1] = mps[i+1] # store the left center tensor
    end
    mPS_canonical[n] = mps[n] # last tensor remains unchanged
    leftcenter[n] = mps[n] # store the last tensor as the left center tensor
    return [mPS_canonical, leftcenter]
end

"""
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
        A = reshape(C, (D, d*h))
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

function canonicalQR(Mps, side)

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
