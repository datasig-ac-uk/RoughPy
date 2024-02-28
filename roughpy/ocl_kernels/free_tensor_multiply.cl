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
// Created by user on 05/09/23.
//

#include "kernel_types.h"
#include "tensor_index_functions.h"

#define identity(x) (x)
#define uminus(x) -(x)
#define post_multiply(x) (x) * alpha
#define TILE_WIDTH levels[tile_letters]
#define MID_STRIDE levels[mid_deg]
#define OFFSET(L, I) offsets[L] + I



#define FTM_TILED_BODY(TYPE, OP)                                               \
    const int mid_deg = out_degree - 2 * tile_letters;                         \
    const size_t bx = get_local_size(0);                                       \
    const size_t by = get_local_size(1);                                       \
    const size_t num_blocks = get_num_groups(0);                               \
    size_t mid_idx = get_group_id(0);                                          \
    size_t work_offsetl, work_offsetr;                                         \
    int lhs_deg = lhs_min_degree;                                              \
    int lhs_deg_max = min(out_degree, lhs_degree);                             \
    int rhs_deg;                                                               \
                                                                               \
    TYPE lhs_val;                                                              \
    TYPE rhs_val;                                                              \
    TYPE out_val;                                                              \
                                                                               \
    for (; mid_idx < MID_STRIDE; mid_idx += num_blocks) {                      \
        for (size_t ix = get_local_id(0); ix < TILE_WIDTH; ix += bx) {         \
            for (size_t iy = get_local_id(1); iy < TILE_WIDTH; iy += by) {     \
                out_val = 0;                                                   \
                                                                               \
                if (lhs_min_degree == 0 && *lhs_src != 0) {                    \
                    work_offsetr = OFFSET(                                     \
                            out_degree,                                        \
                            (ix * MID_STRIDE + mid_idx) * TILE_WIDTH + iy      \
                    );                                                         \
                    out_val += OP(lhs_src[work_offsetr] * (*rhs_src));         \
                }                                                              \
                                                                               \
                for (; lhs_deg < min(tile_letters, lhs_deg_max); ++lhs_deg) {  \
                    rhs_deg = out_degree - lhs_deg;                            \
                                                                               \
                    work_offsetl = split_index(                                \
                            levels[lhs_deg - tile_letters], ix, &work_offsetr  \
                    );                                                         \
                    lhs_val = lhs_src[OFFSET(lhs_deg, work_offsetl)];          \
                    rhs_val = rhs_src[OFFSET(                                  \
                            rhs_deg,                                           \
                            (work_offsetr * MID_STRIDE + mid_idx) * TILE_WIDTH \
                                    + iy                                       \
                    )];                                                        \
                    out_val += OP(lhs_val * rhs_val);                          \
                }                                                              \
                                                                               \
                for (; lhs_deg < min(out_degree - tile_letters, lhs_deg_max);  \
                     ++lhs_deg) {                                              \
                    rhs_deg = out_degree - lhs_deg;                            \
                                                                               \
                    work_offsetl = split_index(                                \
                            levels[rhs_deg - tile_letters], mid_idx,           \
                            &work_offsetr                                      \
                    );                                                         \
                    lhs_val = lhs_src[OFFSET(                                  \
                            lhs_deg,                                           \
                            ix * levels[lhs_deg - tile_letters] + work_offsetl \
                    )];                                                        \
                    rhs_val = rhs_src[OFFSET(                                  \
                            rhs_deg, work_offsetr * TILE_WIDTH + iy            \
                    )];                                                        \
                    out_val += OP(lhs_val * rhs_val);                          \
                }                                                              \
                                                                               \
                for (; lhs_deg < lhs_deg_max; ++lhs_deg) {                     \
                    rhs_deg = out_degree - lhs_deg;                            \
                                                                               \
                    work_offsetl                                               \
                            = split_index(levels[rhs_deg], iy, &work_offsetr); \
                    lhs_val = lhs_src[OFFSET(                                  \
                            lhs_deg,                                           \
                            (ix * MID_STRIDE + mid_idx)                        \
                                            * levels[tile_letters - rhs_deg]   \
                                    + work_offsetl                             \
                    )];                                                        \
                    rhs_val = rhs_src[OFFSET(rhs_deg, work_offsetr)];          \
                    out_val += OP(lhs_val * rhs_val);                          \
                }                                                              \
                                                                               \
                if (lhs_degree >= out_degree && *rhs_src != 0) {               \
                    work_offsetl = OFFSET(                                     \
                            out_degree,                                        \
                            (ix * MID_STRIDE + mid_idx) * TILE_WIDTH + iy      \
                    );                                                         \
                    out_val += OP((*lhs_src) * rhs_src[work_offsetl]);         \
                }                                                              \
                                                                               \
                dst[OFFSET(                                                    \
                        out_degree,                                            \
                        (ix * MID_STRIDE + mid_idx) * TILE_WIDTH + iy          \
                )] += out_val;                                                 \
            }                                                                  \
        }                                                                      \
    }

#define MAKE_FTM_TILED_LEVEL(TYPE, OP)                                         \
    RPY_KERNEL void free_tensor_multiply_tiled_level_##TYPE##_##OP(            \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* lhs_src,      \
            RPY_ADDR_GLOBL const TYPE* rhs_src,                                \
            RPY_ADDR_CONST const size_t* levels,                               \
            RPY_ADDR_CONST const size_t* offsets, int out_degree,              \
            int lhs_degree, int lhs_min_degree, int tile_letters               \
    )                                                                          \
    {                                                                          \
        FTM_TILED_BODY(TYPE, OP)                                               \
    }

MAKE_FTM_TILED_LEVEL(float, identity)
MAKE_FTM_TILED_LEVEL(float, uminus)
MAKE_FTM_TILED_LEVEL(double, identity)
MAKE_FTM_TILED_LEVEL(double, uminus)

// MAKE_FTM_TILED_LEVEL(half, identity)
// MAKE_FTM_TILED_LEVEL(half, uminus)

#undef MAKE_FTM_TILED_LEVEL

#define MAKE_FTM_TILED_LEVEL(TYPE, OP)                                         \
    RPY_KERNEL void free_tensor_multiply_tiled_level_##TYPE##_##OP(            \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* lhs_src,      \
            RPY_ADDR_GLOBL const TYPE* rhs_src,                                \
            RPY_ADDR_CONST const size_t* levels,                               \
            RPY_ADDR_CONST const size_t* offsets, int out_degree,              \
            int lhs_degree, int lhs_min_degree, int tile_letters,              \
            const TYPE alpha                                                   \
    )                                                                          \
    {                                                                          \
        FTM_TILED_BODY(TYPE, OP)                                               \
    }

MAKE_FTM_TILED_LEVEL(float, post_multiply)
MAKE_FTM_TILED_LEVEL(double, post_multiply)

// MAKE_FTM_TILED_LEVEL(half, post_multiply)

#undef MAKE_FTM_TILED_LEVEL
#undef FTM_TILED_BODY

#define FTM_INPLACE_TILED_BODY(TYPE, OP)                                       \
    const int mid_deg = out_degree - 2 * tile_letters;                         \
    const size_t bx = get_local_size(0);                                       \
    const size_t by = get_local_size(1);                                       \
    const size_t num_blocks = get_num_groups(0);                               \
    size_t mid_idx = get_group_id(0);                                          \
    size_t work_offsetl, work_offsetr;                                         \
    int lhs_deg = lhs_min_degree;                                              \
    int lhs_deg_max = min(out_degree, lhs_degree);                             \
    int rhs_deg;                                                               \
                                                                               \
    TYPE lhs_val;                                                              \
    TYPE rhs_val;                                                              \
    TYPE out_val;                                                              \
                                                                               \
    for (; mid_idx < MID_STRIDE; mid_idx += num_blocks) {                      \
        for (size_t ix = get_local_id(0); ix < TILE_WIDTH; ix += bx) {         \
            for (size_t iy = get_local_id(1); iy < TILE_WIDTH; iy += by) {     \
                out_val = 0;                                                   \
                                                                               \
                for (; lhs_deg < min(tile_letters, lhs_deg_max); ++lhs_deg) {  \
                    rhs_deg = out_degree - lhs_deg;                            \
                                                                               \
                    work_offsetl = split_index(                                \
                            levels[lhs_deg - tile_letters], ix, &work_offsetr  \
                    );                                                         \
                    lhs_val = dst[OFFSET(lhs_deg, work_offsetl)];              \
                    rhs_val = rhs_src[OFFSET(                                  \
                            rhs_deg,                                           \
                            (work_offsetr * MID_STRIDE + mid_idx) * TILE_WIDTH \
                                    + iy                                       \
                    )];                                                        \
                    out_val += OP(lhs_val * rhs_val);                          \
                }                                                              \
                                                                               \
                for (; lhs_deg < min(out_degree - tile_letters, lhs_deg_max);  \
                     ++lhs_deg) {                                              \
                    rhs_deg = out_degree - lhs_deg;                            \
                                                                               \
                    work_offsetl = split_index(                                \
                            levels[rhs_deg - tile_letters], mid_idx,           \
                            &work_offsetr                                      \
                    );                                                         \
                    lhs_val = dst[OFFSET(                                      \
                            lhs_deg,                                           \
                            ix * levels[lhs_deg - tile_letters] + work_offsetl \
                    )];                                                        \
                    rhs_val = rhs_src[OFFSET(                                  \
                            rhs_deg, work_offsetr * TILE_WIDTH + iy            \
                    )];                                                        \
                    out_val += OP(lhs_val * rhs_val);                          \
                }                                                              \
                                                                               \
                for (; lhs_deg < lhs_deg_max; ++lhs_deg) {                     \
                    rhs_deg = out_degree - lhs_deg;                            \
                                                                               \
                    work_offsetl                                               \
                            = split_index(levels[rhs_deg], iy, &work_offsetr); \
                    lhs_val = dst[OFFSET(                                      \
                            lhs_deg,                                           \
                            (ix * MID_STRIDE + mid_idx)                        \
                                            * levels[tile_letters - rhs_deg]   \
                                    + work_offsetl                             \
                    )];                                                        \
                    rhs_val = rhs_src[OFFSET(rhs_deg, work_offsetr)];          \
                    out_val += OP(lhs_val * rhs_val);                          \
                }                                                              \
                                                                               \
                work_offsetl = OFFSET(                                         \
                        out_degree,                                            \
                        (ix * MID_STRIDE + mid_idx) * TILE_WIDTH + iy          \
                );                                                             \
                                                                               \
                if (lhs_min_degree == 0 && *dst != 0) {                        \
                    out_val += OP((*dst) * rhs_src[work_offsetl]);             \
                }                                                              \
                                                                               \
                if (lhs_degree >= out_degree && *rhs_src != 0) {               \
                    out_val += OP(dst[work_offsetl] * (*rhs_src));             \
                }                                                              \
                                                                               \
                dst[work_offsetl] = out_val;                                   \
            }                                                                  \
        }                                                                      \
    }

#define MAKE_FTM_INPLACE_TILED_LEVEL(TYPE, OP)                                 \
    RPY_KERNEL void inplace_free_tensor_multiply_tiled_level_##TYPE##_##OP(    \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* rhs_src,      \
            RPY_ADDR_CONST const size_t* levels,                               \
            RPY_ADDR_CONST const size_t* offsets, int out_degree,              \
            int lhs_degree, int lhs_min_degree, int tile_letters               \
    )                                                                          \
    {                                                                          \
        FTM_INPLACE_TILED_BODY(TYPE, OP)                                       \
    }

MAKE_FTM_INPLACE_TILED_LEVEL(float, identity)
MAKE_FTM_INPLACE_TILED_LEVEL(float, uminus)
MAKE_FTM_INPLACE_TILED_LEVEL(double, identity)
MAKE_FTM_INPLACE_TILED_LEVEL(double, uminus)

#undef MAKE_FTM_INPLACE_TILED_LEVEL

#define MAKE_FTM_INPLACE_TILED_LEVEL(TYPE, OP)                                 \
    RPY_KERNEL void inplace_free_tensor_multiply_tiled_level_##TYPE##_##OP(    \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* rhs_src,      \
            RPY_ADDR_CONST const size_t* levels,                               \
            RPY_ADDR_CONST const size_t* offsets, int out_degree,              \
            int lhs_degree, int lhs_min_degree, int tile_letters,              \
            const TYPE alpha                                                   \
    )                                                                          \
    {                                                                          \
        FTM_INPLACE_TILED_BODY(TYPE, OP)                                       \
    }

MAKE_FTM_INPLACE_TILED_LEVEL(float, post_multiply)
MAKE_FTM_INPLACE_TILED_LEVEL(double, post_multiply)

#undef MAKE_FTM_INPLACE_TILED_LEVEL

#define FTM_LEVEL01_IMPL(TYPE, OP)                                             \
    const size_t idx = get_global_id(0);                                       \
                                                                               \
    TYPE out_val = 0;                                                          \
    if (idx == 0 && lhs_min_degree == 0) {                                     \
        out_val += OP((*lhs_src) * (*rhs_src));                                \
    } else {                                                                   \
        if (lhs_min_degree == 0) { out_val += OP((*lhs_src) * rhs_src[idx]); } \
        if (lhs_degree >= 1) { out_val += OP(lhs_src[idx] * (*rhs_src)); }     \
    }                                                                          \
                                                                               \
    dst[idx] += out_val;

#define MAKE_FTM_LEVEL01(TYPE, OP)                                             \
    RPY_KERNEL void free_tensor_multiply_level01_##TYPE##_##OP(                \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* lhs_src,      \
            RPY_ADDR_GLOBL const TYPE* rhs_src, const int lhs_degree,          \
            const int lhs_min_degree                                           \
    )                                                                          \
    {                                                                          \
        FTM_LEVEL01_IMPL(TYPE, OP)                                             \
    }

MAKE_FTM_LEVEL01(float, identity)
MAKE_FTM_LEVEL01(float, uminus)
MAKE_FTM_LEVEL01(double, identity)
MAKE_FTM_LEVEL01(double, uminus)

#undef MAKE_FTM_LEVEL01

#define MAKE_FTM_LEVEL01(TYPE, OP)                                             \
    RPY_KERNEL void free_tensor_multiply_level01_##TYPE##_##OP(                \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* lhs_src,      \
            RPY_ADDR_GLOBL const TYPE* rhs_src, const int lhs_degree,          \
            const int lhs_min_degree, const TYPE alpha                         \
    )                                                                          \
    {                                                                          \
        FTM_LEVEL01_IMPL(TYPE, OP)                                             \
    }

MAKE_FTM_LEVEL01(float, post_multiply)
MAKE_FTM_LEVEL01(double, post_multiply)

#undef MAKE_FTM_LEVEL01

#define FTM_INPLACE_LEVEL01_IMPL(TYPE, OP)                                     \
    const size_t idx = get_global_id(0);                                       \
                                                                               \
    TYPE out_val = 0;                                                          \
    if (idx == 0 && lhs_min_degree == 0) {                                     \
        out_val += OP((*dst) * (*rhs_src));                                    \
    } else {                                                                   \
        if (lhs_min_degree == 0) { out_val += OP((*dst) * rhs_src[idx]); }     \
        if (lhs_degree >= 1) { out_val += OP(dst[idx] * (*rhs_src)); }         \
    }                                                                          \
                                                                               \
    dst[idx] = out_val;

#define MAKE_FTM_INPLACE_LEVEL01(TYPE, OP)                                     \
    RPY_KERNEL void inplace_free_tensor_multiply_level01_##TYPE##_##OP(        \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* rhs_src,      \
            const int lhs_degree, const int lhs_min_degree                     \
    )                                                                          \
    {                                                                          \
        FTM_INPLACE_LEVEL01_IMPL(TYPE, OP)                                     \
    }

MAKE_FTM_INPLACE_LEVEL01(float, identity)
MAKE_FTM_INPLACE_LEVEL01(float, uminus)
MAKE_FTM_INPLACE_LEVEL01(double, identity)
MAKE_FTM_INPLACE_LEVEL01(double, uminus)

#undef MAKE_FTM_INPLACE_LEVEL01

#define MAKE_FTM_INPLACE_LEVEL01(TYPE, OP)                                     \
    RPY_KERNEL void inplace_free_tensor_multiply_level01_##TYPE##_##OP(        \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* rhs_src,      \
            const int lhs_degree, const int lhs_min_degree, const TYPE alpha   \
    )                                                                          \
    {                                                                          \
        FTM_INPLACE_LEVEL01_IMPL(TYPE, OP)                                     \
    }

MAKE_FTM_INPLACE_LEVEL01(float, post_multiply)
MAKE_FTM_INPLACE_LEVEL01(double, post_multiply)

#undef MAKE_FTM_INPLACE_LEVEL01

#undef TILE_WIDTH
#undef MID_STRIDE
#undef OFFSET

#undef identity
#undef uminus
#undef post_multiply
