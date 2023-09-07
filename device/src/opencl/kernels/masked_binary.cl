// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 04/09/23.
//

#include "kernel_types.h"

#define DO_MASK_LOOP(DST, LSRC, RSRC, OP)                                      \
    for (; local_idx < RPY_MASK_ELT_SIZE; local_idx += local_size) {           \
        if ((mask[mask_idx] & (1 << local_idx)) > 0) {                         \
            DST[global_idx] = LSRC[global_idx] OP RSRC[global_idx];            \
        }                                                                      \
    }

#define DO_UNMASK_LOOP(DST, LSRC, RSRC, OP)                                    \
    DST[global_idx] = LSRC[global_idx] OP RSRC[global_idx];

#define MAKE_KERNEL(NAME, OP, TYPE)                                            \
    RPY_KERNEL void masked_##NAME##_##TYPE(                                    \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* lhs_src,      \
            RPY_ADDR_GLOBL const TYPE* rhs_src,                                \
            RPY_ADDR_GLOBL const mask_kt* mask                                 \
    )                                                                          \
    {                                                                          \
        size_t global_idx = get_global_id(0);                                  \
                                                                               \
        if (mask != NULL) {                                                    \
            size_t local_idx = get_local_id(0);                                \
            const size_t local_size = get_local_size(0);                       \
            const mask_kt mask_idx = global_idx / RPY_MASK_ELT_SIZE;           \
                                                                               \
            if (lhs_src != NULL && rhs_src != NULL) {                          \
                DO_MASK_LOOP(dst, lhs_src, rhs_src, OP)                        \
            } else if (lhs_src != NULL) {                                      \
                DO_MASK_LOOP(dst, lhs_src, dst, OP)                            \
            } else if (rhs_src != NULL) {                                      \
                DO_MASK_LOOP(dst, dst, rhs_src, OP)                            \
            }                                                                  \
        } else {                                                               \
            if (lhs_src != NULL && rhs_src != NULL) {                          \
                DO_UNMASK_LOOP(dst, lhs_src, rhs_src, OP)                      \
            } else if (lhs_src != NULL) {                                      \
                DO_UNMASK_LOOP(dst, lhs_src, dst, OP)                          \
            } else if (rhs_src != NULL) {                                      \
                DO_UNMASK_LOOP(dst, dst, lhs_src, OP)                          \
            }                                                                  \
        }                                                                      \
    }

MAKE_KERNEL(add, +, float)
MAKE_KERNEL(add, +, double)

MAKE_KERNEL(sub, -, float)
MAKE_KERNEL(sub, -, double)

MAKE_KERNEL(mul, *, float)
MAKE_KERNEL(mul, *, double)

// TODO: Maybe replace these using intrinsics
MAKE_KERNEL(div, /, float)
MAKE_KERNEL(div, /, double)

#undef DO_MASK_LOOP
#undef DO_UNMASK_LOOP
#undef MAKE_KERNEL

#define DO_MASK_LOOP(DST, LSRC, RSRC, OP)                                      \
    for (; local_idx < RPY_MASK_ELT_SIZE; local_idx += local_size) {           \
        if ((mask[mask_idx] & (1 << local_idx)) > 0) {                         \
            DST[global_idx] = OP(LSRC[global_idx], RSRC[global_idx]);          \
        }                                                                      \
    }

#define DO_UNMASK_LOOP(DST, LSRC, RSRC, OP)                                    \
    DST[global_idx] = OP(LSRC[global_idx], RSRC[global_idx]);

#define MAKE_KERNEL_FN(NAME, OP, TYPE)                                         \
    RPY_KERNEL void masked_##NAME##_##TYPE(                                    \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* lhs_src,      \
            RPY_ADDR_GLOBL const TYPE* rhs_src,                                \
            RPY_ADDR_GLOBL const mask_kt* mask                                 \
    )                                                                          \
    {                                                                          \
        size_t global_idx = get_global_id(0);                                  \
                                                                               \
        if (mask != NULL) {                                                    \
            size_t local_idx = get_local_id(0);                                \
            const size_t local_size = get_local_size(0);                       \
            const mask_kt mask_idx = global_idx / RPY_MASK_ELT_SIZE;           \
                                                                               \
            if (lhs_src != NULL && rhs_src != NULL) {                          \
                DO_MASK_LOOP(dst, lhs_src, rhs_src, OP)                        \
            } else if (lhs_src != NULL) {                                      \
                DO_MASK_LOOP(dst, lhs_src, dst, OP)                            \
            } else if (rhs_src != NULL) {                                      \
                DO_MASK_LOOP(dst, dst, rhs_src, OP)                            \
            }                                                                  \
        } else {                                                               \
            if (lhs_src != NULL && rhs_src != NULL) {                          \
                DO_UNMASK_LOOP(dst, lhs_src, rhs_src, OP)                      \
            } else if (lhs_src != NULL) {                                      \
                DO_UNMASK_LOOP(dst, lhs_src, dst, OP)                          \
            } else if (rhs_src != NULL) {                                      \
                DO_UNMASK_LOOP(dst, dst, rhs_src, OP)                          \
            }                                                                  \
        }                                                                      \
    }

MAKE_KERNEL_FN(min, fmin, float)
MAKE_KERNEL_FN(min, fmin, double)

MAKE_KERNEL_FN(max, fmax, float)
MAKE_KERNEL_FN(max, fmax, double)

#undef DO_MASK_LOOP
#undef DO_UNMASK_LOOP
#undef MAKE_KERNEL_FN

/*
 * Next are the fused multiply-add operations z = a*x + y and z = a*x + b*y
 */

#define DO_MASK_LOOP(DST, LSRC, RSRC, A, OP)                                   \
    for (; local_idx < RPY_MASK_ELT_SIZE; local_idx += local_size) {           \
        if ((mask[mask_idx] & (1 << local_idx)) > 0) {                         \
            DST[global_idx] = OP(A, LSRC[global_idx], RSRC[global_idx]);       \
        }                                                                      \
    }

#define DO_UNMASK_LOOP(DST, LSRC, RSRC, A, OP)                                 \
    DST[global_idx] = OP(A, LSRC[global_idx], RSRC[global_idx]);

#define MAKE_KERNEL_FMA(NAME, TYPE)                                            \
    RPY_KERNEL void masked_##NAME##_##TYPE(                                    \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* lhs_src,      \
            RPY_ADDR_GLOBL const TYPE* rhs_src, const TYPE alpha,              \
            RPY_ADDR_GLOBL const mask_kt* mask                                 \
    )                                                                          \
    {                                                                          \
        size_t global_idx = get_global_id(0);                                  \
                                                                               \
        if (mask != NULL) {                                                    \
            size_t local_idx = get_local_id(0);                                \
            const size_t local_size = get_local_size(0);                       \
            const mask_kt mask_idx = global_idx / RPY_MASK_ELT_SIZE;           \
                                                                               \
            if (lhs_src != NULL && rhs_src != NULL) {                          \
                DO_MASK_LOOP(dst, lhs_src, rhs_src, alpha, TYPE##_fma)         \
            } else if (lhs_src != NULL) {                                      \
                DO_MASK_LOOP(dst, lhs_src, dst, alpha, TYPE##_fma)             \
            } else if (rhs_src != NULL) {                                      \
                DO_MASK_LOOP(dst, dst, rhs_src, alpha, TYPE##_fma)             \
            }                                                                  \
        } else {                                                               \
            if (lhs_src != NULL && rhs_src != NULL) {                          \
                DO_UNMASK_LOOP(dst, lhs_src, rhs_src, alpha, TYPE##_fma)       \
            } else if (lhs_src != NULL) {                                      \
                DO_UNMASK_LOOP(dst, lhs_src, dst, alpha, TYPE##_fma)           \
            } else if (rhs_src != NULL) {                                      \
                DO_UNMASK_LOOP(dst, dst, rhs_src, alpha, TYPE##_fma)           \
            }                                                                  \
        }                                                                      \
    }

MAKE_KERNEL_FMA(axpy, float)
MAKE_KERNEL_FMA(axpy, double)

#undef DO_MASK_LOOP
#undef DO_UNMASK_LOOP
#undef MAKE_KERNEL_FMA

#define DO_MASK_LOOP(DST, LSRC, RSRC, A, B, OP)                                \
    for (; local_idx < RPY_MASK_ELT_SIZE; local_idx += local_size) {           \
        if ((mask[mask_idx] & (1 << local_idx)) > 0) {                         \
            tmp = B * RSRC[global_idx];                                        \
            DST[global_idx] = OP(A, LSRC[global_idx], tmp);                    \
        }                                                                      \
    }

#define DO_UNMASK_LOOP(DST, LSRC, RSRC, A, B, OP)                              \
    tmp = B * RSRC[global_idx];                                                \
    DST[global_idx] = OP(A, LSRC[global_idx], tmp);

#define MAKE_KERNEL_AXPBY(TYPE)                                                \
    RPY_KERNEL void masked_axpby##_##TYPE(                                     \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* lhs_src,      \
            RPY_ADDR_GLOBL const TYPE* rhs_src, const TYPE alpha,              \
            const TYPE beta, RPY_ADDR_GLOBL const mask_kt* mask                \
    )                                                                          \
    {                                                                          \
        float tmp;                                                             \
        size_t global_idx = get_global_id(0);                                  \
                                                                               \
        if (mask != NULL) {                                                    \
            size_t local_idx = get_local_id(0);                                \
            const size_t local_size = get_local_size(0);                       \
            const mask_kt mask_idx = global_idx / RPY_MASK_ELT_SIZE;           \
                                                                               \
            if (lhs_src != NULL && rhs_src != NULL) {                          \
                DO_MASK_LOOP(dst, lhs_src, rhs_src, alpha, beta, TYPE##_fma)   \
            } else if (lhs_src != NULL) {                                      \
                DO_MASK_LOOP(dst, lhs_src, dst, alpha, beta, TYPE##_fma)       \
            } else if (rhs_src != NULL) {                                      \
                DO_MASK_LOOP(dst, dst, rhs_src, alpha, beta, TYPE##_fma)       \
            }                                                                  \
        } else {                                                               \
            if (lhs_src != NULL && rhs_src != NULL) {                          \
                DO_UNMASK_LOOP(dst, lhs_src, rhs_src, alpha, beta, TYPE##_fma) \
            } else if (lhs_src != NULL) {                                      \
                DO_UNMASK_LOOP(dst, lhs_src, dst, alpha, beta, TYPE##_fma)     \
            } else if (rhs_src != NULL) {                                      \
                DO_UNMASK_LOOP(dst, dst, rhs_src, alpha, beta, TYPE##_fma)     \
            }                                                                  \
        }                                                                      \
    }

MAKE_KERNEL_AXPBY(float)
MAKE_KERNEL_AXPBY(double)

#undef DO_MASK_LOOP
#undef DO_UNMASK_LOOP
#undef MAKE_KERNEL_AXPBY
