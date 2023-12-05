# OpenCL kernels for RoughPy
This folder contains all of the built-in kernel definitions that we need to 
implement the standard operations in RoughPy.
These come in two flavours.
The first kind are general purpose kernels for unary or binary vector 
operations 
with optional masking.
The second kind are kernels that perform part of an algebraic operation such 
as free tensor multiplication.
The use of these kernels is generally pretty complicated, so this document 
explains how each of the kernels should be used and what arguments to provide.


## Masked unary kernels
The naming convention is as follows:
```
masked_<operation>_<scalar type>
```
where `<operation>` is one of `uminus` and `scalar_multiply`, and `<scalar 
type>` is one of `float` or `double`.

### Uminus
The unary minus kernel takes two pointers (in the global address space) to 
`<scalar type>` (destination vector and source vector) and one pointer to 
the bitmask vector.
For example:
```c
RPY_KERNEL void masked_uminus_float(
        RPY_ADDR_GLOBL float* dst,
        RPY_ADDR_GLOBL const float* src,
        RPY_ADDR_GLOBL const mask_kt* mask
        )
```
The first two arguments must be non-null pointers to valid memory whose size 
is at least the `get_global_size(0)`.
The third argument can be null, in which case no masking occurs.
If the mask is null then the global size launch parameter should be simply 
the size of the vectors.

If a masking operation is required, the local group size should be equal to 
32 (`sizeof(mask_kt)*8`), and the global work group size should be the 
smallest multiple of 32 larger than the size of the vector (i.e. `32*((size + 
31)/32)`).
The `i`th thread of the work group will take the `i`th bit of `mask
[global_idx/32]` as the indicator of whether the operation should be performed.
The mask bits should be set to 0 for any overflow between vector size and 
`32*mask_size`.

### Scalar multiply
The operation is identical to uminus, except for the addition of the scalar 
multiplier parameter:
```c
RPY_KERNEL void masked_scalar_multiply_float(
        RPY_ADDR_GLOBL float* dst,
        RPY_ADDR_GLOBL const float* src,
        const float scalar,
        RPY_ADDR_GLOBL const mask_kt* mask,
        )
```


## Masked binary kernels
The masked binary operator kernels have a similar naming convention as masked 
unary kernels:
```
masked_<name>_<scalar type>
```
where `<name>` is the full name of the operation (e.g. `add`, `sub`, `mul`, 
`axpy` etc.) and, as before, `<scalar type>` is one of `float` or `double`.

As before, the mask argument is optional and can be null to indicate that 
the operation should be performed on all elements.
At least one of the left hand side or right hand side points must be valid 
and the destination pointer should point to valid memory.

### Add, Sub, Mul, and Div
These four operations implement vector addition, subtraction, and 
coordinate-wise multiplication and division.
The signature is as follows:
```c
RPY_KERNEL void masked_add_float(
        RPY_ADDR_GLOBL float* dst,
        RPY_ADDR_GLOBL const float* lhs_src,
        RPY_ADDR_GLOBL const float* rhs_src,
        RPY_ADDR_GLOBL const mask_kt* mask
        )
```
If both `lhs_src` and `rhs_src` are non-null, and no masking is set, then the 
result at index `i` 
will be
```c
dst[i] = lhs_src[i] + rhs_src[i]
```
However, if `lhs_src` or `rhs_src` is null then the operation is performed 
in-place taking `dst` to be the missing vector.
For instance, if `lhs_src` is null, then the operation is
```c
dst[i] += rhs_src[i]
```
In all three of these unmasked cases, the global work size should be equal 
to the size of the vector.
