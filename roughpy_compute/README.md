# RoughPy compute

This component contains all the basic algebra operations that are required for computational rough paths. Currently all 
the operations are implemented via the corresponding algebra instance (mostly through libalgebra-lite). The plan is to
eventually remove libalgebra-lite in favour of a more general framework for implementing functions as free functions, 
which should greatly improve the flexibility. This will eventually replace all the internal implementations of algebra
operations as well as provide a base for standalone computation libraries alongside.

The design goal is quite simple. We need to provide implementations of the core operations, described below, for each 
of the ways that we might "see" an instance of an algebra. More on this later. This is a short-term solution. In the 
long-term, we can build out a more sophisticated framework for building these operations that will be able to unify all
these "views". But, before we can do any of that, we need a basic implementation that we can a) test against, and b) use
to understand the requirements.

## Operations

We have a relatively small number of basic operations from which all operations are derived. More information about each
operation is given below. The basic operations are:

 - basic vector operations (addition and scalar multiplication, and the inplace versions)
 - free tensor multiplication (inplace and fused multiply-add)
 - free tensor antipode (generalized transpose)
 - Lie-to-tensor and tensor-to-Lie sparse linear maps
 - left and right half-shuffles, and the full shuffle product
 - the adjoint of the left multiplier operator on the free tensor algebra
 - the adjoint of the (left) multiplier operator on the shuffle tensor algebra
 - width increase or decrease operators on two tensor algebras

On top of these, we can derive a number of intermediate operations:

 - free tensor exponential and fused multiply-exponential (fmexp)
 - free tensor logarithm
 - the adjoint of the right multiplier operator on the free tensor algebra
 - Lie multiplication
 - change of Lie basis

The basic operations are where we need to spend the most time getting these right, since all other operations build on 
top of them. This will require some rigorous testing, probably making use of polynomial coefficients, to ensure 
numerical correctness.





## Context

This is a very brief introduction to signatures in the context of data analysis. There are many other resources out
there that give more information about this. The [RoughPy paper](https://proceedings.scipy.org/articles/DXWY3560) a 
short but very reasonable introduction and contains many references to what I consider to be some of the most 
illustrative examples.

Signatures are a high-order summary of the evolution of a path. Given a path $\gamma:[0, T]$ the signature of $\gamma$,
over some subinterval $[s, t]$ is a free tensor $S(\gamma)_{[s,t]}$ formally defined by iterated integrals of $\gamma$.
Signatures have the property that if we concatenate two paths (or, for convenience, consider the signature of $\gamma$ 
over adjacent intervals) then the signature of the concatenation is the tensor product of the signatures; that is,
$S(\gamma)_{[s, u]} = S(\gamma)_{[s, t]} \otimes S(\gamma)_{[t, u]}$. This is called _Chen's identity_. 

Of course, the signature itself is a full, infinite free tensor. So for practical, computational tasks we have to
truncate at some particular degree, which we call the depth. Truncated signatures are denoted $S^n(\gamma)_{[s, t]}$.
Now if we assume that $\gamma$ is linear on $[s,t]$, then we can signature can be computed explicitly as the tensor
exponential of the "increment" $\gamma(t) - \gamma(s)$ appropriately embedded into the tensor algebra. (More on this
in a moment.) Thus $S(\gamma)_{[s, t]} = \exp(\gamma(t) - \gamma(s))$. The wonderful thing about the tensor exponent and
truncation is that the computation is exact after $n$ computations, provided that the exponent has 0 in the degree 0 
position. 

The way we embed increments into the tensor algebra is actually via the Lie algebra. The Lie algebra (like the tensor 
algebra) contains $\Bbb R^d$ as a subspace (corresponding to the elements of degree 1). Thus we can embed increments
$\vec x = \gamma(t) - \gamma(s)$ into the Lie algebra and then map these into the tensor algebra via the lie-to-tensor
linear map. This extra steps is instructive. Suppose instead that instead we have a genuine rough path, where the
iterated integrals cannot be computed directly (with sufficient accuracy) from the plain increments; for instance, 
Brownian motion. To propery compute the signature here we would actually need not only the "increment" data, but also
some higher order data (for Brownian motion, the Levy areas) to compute the signature properly. These higher-order terms
live naturally inside the Lie algebra. The process of computing the signature is then exactly the same: embed into the
tensor algebra using lie-to-tensor and exponentiate.

### Piecewise Abelian paths
One very basic instance of a rough path is the rough generalisation of the piecewise-linear path which we call a 
_piecewise Abelian path_. This is defined as follows. Consider a partition $\{0=t_0< t_1< t_2< \dots< t_N=T\}$ of 
$[0, T]$ and suppose that for each of the intervals $[t_{j-1}, t_{j}]$ one has a corresponding element $L_j$ of the 
Lie algebra. If we computed the signature of this path over $[t_{j-1}, t_j]$ the result would be $\exp(L_j)$ (to keep
the notation simple, we omit the lie-to-tensor map here). If we take a sub-interval $[s, t]$ of $[t_{j-1}, t_j]$ then
we have to scale $L_j$ prior to exponentiating, the amount we scale by is the proportion of $[t_{j-1}, t_j]$ that is 
"seen" by $[s, t]$, given by $p = (t - s) / (t_j - t_{j-1})$. Thus the signature is now $\exp(pL_j)$. When the query 
interval $[s, t]$ spans two (or more) partition intervals we have to scale each $L_i$ that appears by the corresponding
proportion $p_i$ of the interval $[t_{i-1}, t_i]$ that is seen by $[s, t]$ and then tensor the exponentials together:
$$
\exp(p_iL_i) \otimes \exp(p_{i+1}L_{i+1}). 
$$
This computation is very cheap consisting of two scalar multiplications and two lie-to-tensor maps, a single tensor
exponential, and a single fused multiply-exponential. The only complicated part is figuring out which of the partition 
intervals are involved in a given computation. From a batching point of view, it might actually be easier to assume that 
all of them are and just compute the proportion as zero in those cases.

One very useful fact is that every path can be turned into a piecewise Abelian path with the provision of a partition
of its support. To do this, one simply computes the log signatures of the path $\delta:[0, T']$ - defined by 
$\log(S(\delta))_{[u, v]}$ (omitting the map from tensor to Lie) - over each of the intervals in the partition. In 
particular, RoughPy can deliver such paths. 




## Basic operations

### Free tensor multiplication

The free tensor product is our most used operation. This is derived from the concatenation product on tensor words
(the keys of the basis for the tensor algebra). Because of a relationship between a tensor word and its index in the 
total order on the basis, it's relatively easy to write down an algorithm for performing a basic tensor multiplication 
expressed below in Python.  Note that the following implementation is horrible for many reasons.

```python
import numpy as np

# For example
width = 2
depth = 5
out = np.zeros(65)
lhs = np.array(...)
rhs = np.array(...)

for lhs_deg in range(depth+1):
    lhs_offset = sum(width**i for i in range(lhs_deg))
    lhs_level = lhs[lhs_offset:lhs_offset + width**lhs_deg]
    
    for rhs_deg in range(depth+1):
        rhs_offset = sum(width**i for i in range(rhs_deg))
        rhs_level = rhs[rhs_offset:rhs_offset + width**rhs_deg]
        
        if lhs_deg + rhs_deg > depth:
            continue
            
        for lhs_idx in range(width ** lhs_deg):
            for rhs_idx in range(width ** rhs_deg):
                out[lhs_idx * (width**rhs_deg) + rhs_idx] += lhs_level[lhs_idx] * rhs_level[rhs_idx]
        
```

There are a great many "easy" optimizations that we can use to implement a basic version of this operation, which should 
be the first thing we implement as our reference. (This slightly improved co-product is the same as that currently 
implemented in libalgebra-lite). 

There is a cache-friendly, tiled algorithm for computing free-tensor products that should greatly improve performance. 
However, the algorithm is sligtly more tricky to implement. This is implemented in libalgebra, but it is very much
obscured by the supporting framework that could surely be simplified.

The inplace version demands a very specific handling because one must be careful to only write to the result once the
index in question will not be used for any future products. (Note that lower degree terms in each operand contribute to
higher degree terms in the product.)

### Antipode

This operation is induced by reversing the tensor word, and generalizes the transpose operation on matrices. In order
to compute this, we do have to convert from index to word and back again, which can be a fairly painful conversion if
done piecemeal. The only complication here is that the terms of degree $d$ are signed by multiplying by $(-1)^d$. 
Ignoring these conversions, the algorithm is fairly simple:

```python
for src_idx in range(tensor_size):
    word = idx_to_word(src_idx)
    degree = len(word)
    rev_word = reversed(word)
    dst_idx = word_to_idx(rev_word)
    
    dst[dst_idx] = (-1)**degree * src[src_idx]
```

There is a much better algorithm that once again makes use of tiles. This is easier to implement compared to the tiled
 multiplication and should probably be our baseline operation. This makes the conversion from letters to words much
less painful as well as improving the cache locality.

### Lie to tensor and tensor to Lie

These are sparse linear maps that embed the Lie algebra into the free tensor algebra and the corresponding projection
from the tensor algebra onto the Lie algebra. The Lie to tensor map recursively expands brackets to tensor commutators.
(letters are mapped identically). For width 2, depth 3 we can easily write down the matrix for this operation.

```python
np.array([[
 #  1, 2, [1, 2], [1, [1, 2]], [2, [1, 2]]
   [1, 0,      0,           0,           0],     # (1)
   [0, 1,      0,           0,           0],     # (2)
   [0, 0,      0,           0,           0],     # (1, 1)
   [0, 0,      1,           0,           0],     # (1, 2)
   [0, 0,     -1,           0,           0],     # (2, 1)
   [0, 0,      0,           0,           0],     # (2, 2)
   [0, 0,      0,           0,           0],     # (1, 1, 1)
   [0, 0,      0,           1,           0],     # (1, 1, 2)
   [0, 0,      0,          -2,           0],     # (1, 2, 1)
   [0, 0,      0,           0,          -1],     # (1, 2, 2)
   [0, 0,      0,           1,           0],     # (2, 1, 1)
   [0, 0,      0,           0,           2],     # (2, 1, 2)
   [0, 0,      0,           0,           1],     # (2, 2, 1)
   [0, 0,      0,           0,           0],     # (2, 2, 2)
]])
```

Notice that this is extremely sparse. We can either write it out in sparse form (csr format is most appropriate for
this application) or we can perform the recursive expansion directly inside the function and bypass the need to 
explicitly construct the matrix.

The tensor to Lie map performs recursive (right) bracketing. Note that this is sligtly more tricky because (1, 2, 1) 
does not expand neatly to a Lie word because [1, [2, 1]] is not an element in our (usual) Hall set. Here we have to use
the identity [u, v] = -[v, u].

These are the only basic operations that involve the Lie algebra, and thus involve a choice of Hall set. We typically 
use a "greedily constructed" Hall set that prioritizes brackets in the right-hand term of a bracket (i.e. 
[1, [2, [1, 3]]] appears before [[1, 2], [1, 3]] in the basis order). This is the one that libalgebra (and 
libalgebra-lite) use, and is generally a fairly sensible choice. Another choice that appears elsewhere in the 
computational rough paths ecosystem is the Lyndon basis. Ideally we should be agnostic to the actual Lie basis used, and
our computations should be able to handle all of these. Fortunately, there is a fairly compact way to describe any Hall
set that is portable and easy to use. This involves two arrays of integers: one containing pairs of "parents" for each
entry of the set and one set of "degree ranges". The latter is a useful construct for all kinds of graded bases. The 
struct I like to work with is defined as follows:

```c++
struct HallSet {
    size_t const * data;    // size 2 * (lie_size + 1) with first element (0, 0)
    size_t const * begin;   // size depth + 2
    int32_t width;          // redunant, because it is begin[1] - begin[0], but useful
    int32_t depth;
}; 
```

This can be silently passed in to any operation that involves a Lie to facilitate computations. 


### Left and right half-shuffle and the shuffle product




### The adjoint of the left multiplier operator on the free tensor algebra

Consider a given free tensor $x\in T((V))$ and denote by $L_x$ the operator on $T((V))$ defined by 
$L_x(y) = x \otimes y$. This is the left multiplier operator. As a linear operator on $T((V))$ there is a corresponding
adjoint operator $L_x^*:T((V))' \to T((V))'$ (which can also be considered as an operator on $T((V))$). This operator
appears in several contexts, notably in the system of PDEs used to compute signature kernels on rough streams. In the
simplified case where $x$ is a single word, $w$, the adjoint operator is that extracts the coefficients of all the words
prefixed by $w$. Thus $y$ maps to $\sum_v \langle y, wv\rangle v$. where $\langle y, wv\rangle$ denotes the coefficient
of $wv$ in $y$. The full operator is the natural linear extension of this simplified case.

The dense version of this is very easy to write down, again because of the relationship between the index of a word in
the basis total order and the word itself. More precisely, if $w$ has index $i$ in the lexicographic order on words of 
length $d := \ell(w)$, then each index of a word with length $k$ prefixed by $w$ has an index given by $iW^{k - d} + j$.
If $x$ is a dense vector, we can tile several prefix words together to maximize the cache locality too.


### The adjoint of multiplication operator on the shuffle algebra

This is the analogous operator but where the free tensor product is replaced by the shuffle product. (Since shuffle is 
commutative, there is only one such operator.) This one I know less about, but is is implemented (in pure Python) in the
"tensor_functions" module of RoughPy, where it is used in the computation of LOG. (The linear extension of the log 
function to the whole tensor algebra.)

### Width increase or decrease operator on the tensor algebras

Consider a vectors space $V$ and a subspace $U$ then the natural inclusion of $U$ into $V$ induces an algebra embedding
of $T((U))$ into $T((V))$. If we suppose that $U$ and $V$ have "compatible" bases, then this map can be realised as a 
map on tensor words. This operation increases the width of the tensor in question in a natural way.

For instance, suppose $U$ has width 3 and V width 5. One possible inclusion operator maps $1\to 1$, $2 \to 2$, and 
$3\to 3$. Another example might map $1 \to 2$, $2 \to 3$ and $3\to 5$. Both can be easily implemented at a word level 
for densely represented tensors by again considering the indices. This operation is essential if we consider 
"width-compressed" sparsity for tensors.

The projection from $V$ onto $U$ is also useful. This is implemented in the completely analogous way.



## Densely stored objects


