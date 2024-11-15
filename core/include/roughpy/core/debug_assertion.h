//
// Created by sam on 13/11/24.
//

#ifndef ROUGHPY_CORE_DEBUG_ASSERTION_H
#define ROUGHPY_CORE_DEBUG_ASSERTION_H

#include <cassert>

#if (defined(_DEBUG) || !defined(NDEBUG) || !defined(__OPTIMIZE__))            \
        && !defined(RPY_DEBUG)
#  define RPY_DEBUG
#elif !defined(RPY_DEBUG)
#  undef RPY_DEBUG
#endif

#ifdef RPY_DEBUG
#  include "check_helpers.h"
#endif

#ifdef RPY_DEBUG
#  if defined(RPY_GCC) || defined(RPY_CLANG)
#    define RPY_DBG_ASSERT(ARG) assert(ARG)
#  else
#    define RPY_DBG_ASSERT(ARG) assert(ARG)
#  endif
#else
#  define RPY_DBG_ASSERT(ARG) (void) 0
#endif

#define RPY_DBG_ASSERT_EQ(a, b) RPY_DBG_ASSERT((::rpy::compare_equal((a), (b))))
#define RPY_DBG_ASSERT_NE(a, b)                                                \
    RPY_DBG_ASSERT((::rpy::compare_not_equal((a), (b))))
#define RPY_DBG_ASSERT_LT(a, b) RPY_DBG_ASSERT((::rpy::compare_less((a), (b))))
#define RPY_DBG_ASSERT_GT(a, b)                                                \
    RPY_DBG_ASSERT((::rpy::compare_greater((a), (b))))
#define RPY_DBG_ASSERT_LE(a, b)                                                \
    RPY_DBG_ASSERT((::rpy::compare_less_equal((a), (b))))
#define RPY_DBG_ASSERT_GE(a, b)                                                \
    RPY_DBG_ASSERT((::rpy::compare_greater_equal((a), (b))))

#endif// ROUGHPY_CORE_DEBUG_ASSERTION_H
