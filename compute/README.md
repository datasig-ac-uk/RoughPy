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
 - change of Lie basis

The basic operations are where we need to spend the most time getting these right, since all other operations build on 
top of them. This will require some rigorous testing, probably making use of polynomial coefficients, to ensure 
numerical correctness.










## More on the operations

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
for idx in range(tensor_size):
    word = idx_to_word(idx)
    degree = len(word)
    rev_word = reversed(word)
    dst_idx = word_to_idx(rev_word)
    
    out[dst_idx] = (-1)**degree * src[idx]
```

There is a much better algorithm that once again makes use of tiles. This is easier to implement compared to the tiled
 multiplication and should probably be our baseline operation. This makes the conversion from letters to words much
less painful as well as improving the cache locality.

### Lie to tensor and tensor to Lie

These are sparse linear maps that embed the Lie algebra into the free tensor algebra and the corresponding projection
from the tensor algebra onto the Lie algebra. The Lie to tensor map recursively expands brackets to tensor commutators.