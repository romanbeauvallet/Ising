#!/usr/bin/env julia

##Librairies
using LinearAlgebra
using ITensors
using TensorCast
using TensorOperations
using Dates

##Functions
include("Codes.jl")

function main()
    println("RUNNING Main.jl at ", Dates.now())
    Mps = MPS(2, 2, 2)
    #println("norm ", normalize(Mps))
    #normal que ça donne pas le meme résultat que normalize car norm n'a pas trop de sens pour un MPS
    println("canonical form", canonicalleft(Mps))
    println("canonical form", canonicalright(Mps))
end

#fonction normalize ne marche pas car résultat différent de norm(Mps), surement à cause des axes il faut prendre le vrai dagger donc permutter les axes aussi 

main()