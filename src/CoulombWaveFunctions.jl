"""
# CoulumbWaveFunctions.jl
A Julia native implementation of coulomb wave functions
"""
module CoulombWaveFunctions

export sphericalbessel_jy, coulomb_H, coulomb_FG

function contfrac_spbesselj(l::Int, x, maxiter)
    α = f = eps(Float64)
    bn = β = Δ = 0.
    @inbounds for n = 1:maxiter
        bn = (2l+1+2n)/x
        α = bn - 1/α #(α==0 ? α=eps(Float64) : );
        β = 1/(bn - β) #(α==0 ? α=eps(Float64) : );
        #β ^=(-1);
        Δ = α*β ; f*=Δ ;
        if abs(Δ-1)<eps(Float64)
            break
        end
    end
    return -f
end


@doc raw"""
    sphericalbessel_jy(l::Int, x; maxiter=10000)

Implementation of spherical bessel functions ``j_l(x),y_l(x)``

## Arguments
* `l`: Integer order
* `x`: radial coordinate

## Return Values
* `j` :  ``j_l(x)``
* `y` :  ``y_l(x)``
* `jp`:  ``\frac{d}{dx}j_l(x)``
* `yp`:  ``\frac{d}{dx}y_l(x)``
"""
function sphericalbessel_jy(l::Int, x; maxiter=10000)
    yo=y=-cos(x)/x; yp=(x*sin(x)+cos(x))/(x^2);
    jo=j=sin(x)/x; jp=(x*cos(x)-sin(x))/(x^2);
    @inbounds for n=0:l-1
        ##################y
        yo = y
        y = yo*(n/x) - yp
        yp = yo - y*(n+2)/x
        ##################j
        jo = j
        if n<x
            j = jo*(n/x) - jp
        else
            j *= contfrac_spbesselj(n,x,maxiter)
        end
        jp = jo - j*(n+2)/x
    end
    return j,y,jp,yp
end

## Implementation of coulomb hankel wave function H+
# coulomb_H(l, η, x)
# args: | l-order | η-charge constant | x-unitless radial coordinate (ρ) |
# output: Hl⁺(η,x) , (d/dx)Hl⁺(η,x)
function contfrac_CF1(l, η, x, maxiter)
    α = f = (l+1)/x + η/(l+1)
    an = bn = β = Δ = 0.
    sg = -1
    @inbounds for lpn = l+1:1:l+maxiter
        an = 1+(η/(lpn))^2
        bn = (1+2lpn) * (1/x + η/(lpn*(lpn+1)))
        α = bn - an/α #(α==0 ? α=eps(Float64) : );
        β = 1/(bn - an*β) #(α==0 ? α=eps(Float64) : );
        Δ = α*β ; f*=Δ
        sg *= sign(β)
        if abs(Δ-1)<eps(Float64)
            break
        end
    end
    return f,sg
end

function contfrac_CF2(l, η, x, maxiter)
    α = f = complex(x-η,0)
    bn = β = Δ = complex(0,0)
    @inbounds for n = 1:maxiter
        an = complex(n-l-1,η) * complex(n+l,η)
        bn = 2.0*complex(x-η,n)
        α = bn + an/α
        β = 1/(bn + an*β)
        Δ = α*β
        f *= Δ ;
        if abs(Δ-1)<eps(Float64)
            break
        end
    end
    return -imag(f)/x, real(f)/x
end

function coulomb_H(l, η, x; maxiter=10000)
    f,sg = contfrac_CF1(l,η,x,maxiter)
    p,q = contfrac_CF2(l,η,x,maxiter)
    γ = (f-p)/q
    F = -sg * sqrt(abs(q))/hypot(q,f-p)
    Fp = f*F
    G = γ*F
    Gp = (p*γ-q)*F
    return complex(G,F), complex(Gp,Fp)
end

function coulomb_FG(l, η, x; maxiter=10000)
    f,sg = contfrac_CF1(l,η,x,maxiter)
    p,q = contfrac_CF2(l,η,x,maxiter)
    γ = (f-p)/q
    F = -sg * sqrt(abs(q))/hypot(q,f-p)
    Fp = f*F
    G = γ*F
    Gp = (p*γ-q)*F
    return F,G,Fp,Gp
end

##  gsl_coulomb_H(l, η::Float64, x::Float64)
#   wapper of Coulomb Wave Functions F,G from GNU Scientific Library(GSL)
#   args: | l-order | η-charge constant | x-unitless radial coordinate (ρ) |
#   output: Hl⁺(η,x) , (d/dx)Hl⁺(η,x) ;    both are ComplexF64
function gsl_coulomb_H(l, η::Float64, x::Float64)
    FG = map(x->x.val,sf_coulomb_wave_FG_e(η,x,l,0,0.,0.))
    return complex(FG[3],FG[1]), complex(FG[4],FG[2])
end

##  spfuncs_spbessel_jy(l,x)
#   wapper of sphericalbesselj/y from SpecialFunctions.jl
#   args: | l-order | x-unitless radial coordinate (ρ) |
#   output: jl, yl, jl', yl'
function spfuncs_spbessel_jy(l,x)
    j=sphericalbesselj(l,x);
    jp=(l*sphericalbesselj(l-1,x)-(l+1)*sphericalbesselj(l+1,x))/(2l+1)
    y=sphericalbessely(l,x)
    yp=(l*sphericalbessely(l-1,x)-(l+1)*sphericalbessely(l+1,x))/(2l+1)
    return j,y,jp,yp
end

end # module

## References：

# [1] High-precision evaluation of the regular and irregular Coulomb wavefunctions A. R. Barnett. Mathematics. 1982.

# [2] Barnett A.R. (1996) The Calculation of Spherical Bessel and Coulomb Functions. In: Bartschat K. (eds) Computational Atomic Physics. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-61010-3_9
