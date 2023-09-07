// copyright (c) 2023 the roughpy developers. all rights reserved.
//
// redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// this software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. in no event shall the copyright holder or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or business
// interruption) however caused and on any theory of liability, whether in
// contract, strict liability, or tort (including negligence or otherwise)
// arising in any way out of the use of this software, even if advised of the
// possibility of such damage.

#include "kernel_types.h"
#include "tensor_index_functions.h"

#define signing(x) (degree & 1) ? -(x) : (x)
#define nonsigning(x) (x)

#define TILE_WIDTH levels[tile_letters]
#define MID_STRIDE levels[mid_deg]

#define MAKE_ANTIPODE_KERNEL_TILED_LEVEL(TYPE, OP)                             \
    RPY_KERNEL void antipode_tiled_level_##TYPE##_##OP(                        \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* src,          \
            RPY_ADDR_CONST const size_t* levels, int degree, int tile_letters  \
    )                                                                          \
    {                                                                          \
        const int mid_deg = degree - 2 * tile_letters;                         \
        size_t mid_idx = get_group_id(0);                                      \
        size_t rmid_idx;                                                       \
        const size_t num_blocks = get_num_groups(0);                           \
        size_t jx, jy;                                                         \
        const size_t bx = get_local_size(0);                                   \
        const size_t by = get_local_size(1);                                   \
                                                                               \
        for (; mid_idx < MID_STRIDE; mid_idx += num_blocks) {                  \
            rmid_idx = reverse_index(mid_deg, levels[1], mid_idx);             \
            for (size_t ix = get_local_id(0); ix < TILE_WIDTH; ix += bx) {     \
                jx = reverse_index(tile_letters, levels[1], ix);               \
                for (size_t iy = get_local_id(0); iy < TILE_WIDTH; iy += by) { \
                    jy = reverse_index(tile_letters, levels[1], iy);           \
                    dst[(jy * MID_STRIDE + rmid_idx) * TILE_WIDTH + jx] = OP(  \
                            src[(ix * MID_STRIDE + mid_idx) * TILE_WIDTH + iy] \
                    );                                                         \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

MAKE_ANTIPODE_KERNEL_TILED_LEVEL(float, signing)
MAKE_ANTIPODE_KERNEL_TILED_LEVEL(float, nonsigning)

MAKE_ANTIPODE_KERNEL_TILED_LEVEL(double, signing)
MAKE_ANTIPODE_KERNEL_TILED_LEVEL(double, nonsigning)

#undef MAKE_ANTIPODE_KERNEL_TILED_LEVEL
#undef signing

#define signing(x) (idx > 0) ? -(x) : (x)

#define MAKE_ANTIPODE_KERNEL_TILED_LEVEL01(TYPE, OP)                           \
    RPY_KERNEL void antipode_tiled_level01_##TYPE##_##OP(                      \
            RPY_ADDR_GLOBL TYPE* dst, RPY_ADDR_GLOBL const TYPE* src          \
    )                                                                          \
    {                                                                          \
        size_t idx = get_global_id(0);                                         \
        dst[idx] = OP(src[idx]);                                               \
    }

MAKE_ANTIPODE_KERNEL_TILED_LEVEL01(float, signing)
MAKE_ANTIPODE_KERNEL_TILED_LEVEL01(float, nonsigning)

MAKE_ANTIPODE_KERNEL_TILED_LEVEL01(double, signing)
MAKE_ANTIPODE_KERNEL_TILED_LEVEL01(double, nonsigning)

#undef signing
#undef nonsigning
#undef MAKE_ANTIPODE_KERNEL_TILED_LEVEL01
#undef TILE_WIDTH
#undef MID_STRIDE