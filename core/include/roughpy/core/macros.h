// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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
// Created by user on 04/03/23.
//

#ifndef ROUGHPY_CORE_MACROS_H
#define ROUGHPY_CORE_MACROS_H

#include <cassert>
#include <stdexcept>


#define RPY_STRINGIFY_IMPL(ARG) #ARG
#define RPY_STRINGIFY(ARG) RPY_STRINGIFY_IMPL(ARG)

#define RPY_JOIN_IMPL_IMPL(LHS, RHS) X##Y
#define RPY_JOIN_IMPL(LHS, RHS) RPY_JOIN_IMPL_IMPL(LHS, RHS)
#define RPY_JOIN(LHS, RHS) RPY_JOIN_IMPL(LHS, RHS)

#define RPY_IDENTITY(ARG) ARG

#if (defined(_DEBUG) || !defined(NDEBUG) || !defined(__OPTIMIZE__)) && !defined(RPY_DEBUG)
#define RPY_DEBUG
#else
#undef RPY_DEBUG
#endif

#if defined(_MSC_VER) && defined(_MSVC_LANG)
#define RPY_CPP_VERSION _MSVC_LANG
#define RPY_MSVC
#else
#define RPY_CPP_VERSION __cplusplus
#undef RPY_MSVC
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define RPY_GCC
#else
#undef RPY_GCC
#endif

#ifdef __clang__
#define RPY_CLANG
#else
#undef RPY_CLANG
#endif

#if defined(__linux__)
#  define RPY_PLATFORM_LINUX 1
#elif defined(__APPLE__) && defined(__MACH__)
#  define RPY_PLATFORM_MACOS 1
#elif defined(_WIN32)
#  define RPY_PLATFORM_WINDOWS 1
#endif


#if defined(RPY_PLATFORM_WINDOWS) || defined(__CYGWIN__)
#define RPY_COMPILING_DLL
#endif

#ifdef RPY_COMPILING_DLL
#  if RPY_BUILDING_LIBRARY
#    if defined(RPY_MSVC)
#      define RPY_EXPORT __declspec(dllexport)
#      define RPY_LOCAL
#    else
#      define RPY_EXPORT __attribute__((dllexport))
#      define RPY_LOCAL
#    endif
#  else
#    if defined(RPY_MSVC)
#      define RPY_EXPORT __declspec(dllimport)
#      define RPY_LOCAL
#    else
#      define RPY_EXPORT __attribute__((dllimport))
#    endif
#  endif
#else
#  if defined(RPY_GCC) || defined (RPY_CLANG)
#    if defined(RPY_BUILDING_LIBRARY)
#      define RPY_EXPORT __attribute__((visibility("default")))
#      define RPY_LOCAL __attribute__((visibility("hidden")))
#    else
#      define RPY_EXPORT __attribute__((visibility("default")))
#      define RPY_LOCAL __attribute__((visibility("hidden")))
#    endif
#  else
#    define RPY_EXPORT
#    define RPY_LOCAL
#  endif
#endif


#if RPY_CPP_VERSION >= 201403L
#define RPY_CPP_14
#else
#undef RPY_CPP_14
#endif

#if RPY_CPP_VERSION >= 201703L
#define RPY_CPP_17
#else
#undef RPY_CPP_17
#endif

#ifdef RPY_CPP_17
#define RPY_UNUSED [[maybe_unused]]
#define RPY_NO_DISCARD [[nodiscard]]
#define RPY_IF_CONSTEXPR constexpr
#else
#define RPY_UNUSED
#define RPY_NO_DISCARD
#define RPY_IF_CONSTEXPR
#endif


#if defined(RPY_GCC) || defined(RPY_CLANG)
#define RPY_UNUSED_VAR __attribute__((unused))
#else
#define RPY_UNUSED_VAR
#endif

#if defined(RPY_GCC) || defined(RPY_CLANG)
#define RPY_UNREACHABLE() (__builtin_unreachable())
#define RPY_UNREACHABLE_RETURN(...) RPY_UNREACHABLE()
#elif defined(RPY_MSVC)
#define RPY_UNREACHABLE() (__assume(false))
#define RPY_UNREACHABLE_RETURN(...) \
    RPY_UNREACHABLE();              \
    return __VA_ARGS__
#else
#define RPY_UNREACHABLE()
#define RPY_UNREACHABLE_RETURN(...) \
    RPY_UNREACHABLE();              \
    return __VA_ARGS__
#endif

// Macros that control optimisations

#if defined(__OPTIMIZE__) || !defined(RPY_DEBUG)
#if defined(_WIN32) || defined(_WIN64)
#define RPY_INLINE_ALWAYS __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define RPY_INLINE_ALWAYS inline __attribute__((always_inline))
#else
#define RPY_INLINE_ALWAYS inline
#endif
#endif

#ifdef RPY_MSVC
#define RPY_INLINE_NEVER __declspec(noinline)
#elif defined(RPY_GCC) || defined(RPY_CLANG)
#define RPY_INLINE_NEVER __attribute__((never_inline))
#else
#define RPY_INLINE_NEVER
#endif

#if defined(RPY_GCC) || defined(RPY_CLANG)
#define RPY_RESTRICT __restrict__
#elif defined(RPY_MSVC)
#define RPY_RESTRICT __restrict
#else
#define RPY_RESTRICT
#endif

#if defined(RPY_GCC) || defined(RPY_CLANG)
#define RPY_LIKELY(COND) (__builtin_expect(static_cast<bool>(COND), 1))
#define RPY_UNLIKELY(COND) (__builtin_expect(static_cast<bool>(COND), 0))
#else
#define RPY_LIKEY(COND) (COND)
#define RPY_UNLIKELY(COND) (COND)
#endif

#define RPY_CHECK(EXPR)                                                              \
    do {                                                                             \
        if (RPY_UNLIKELY(!(EXPR))) {                                                 \
            throw std::runtime_error(std::string("failed check \"") + #EXPR + "\""); \
        }                                                                            \
    } while (0)

#ifdef RPY_DEBUG
#ifdef RPY_DBG_ASSERT_USE_EXCEPTIONS
#define RPY_DBG_ASSERT(ARG)                                                                    \
    do {                                                                                       \
        if (RPY_UNLIKEY(!(EXPR))) {                                                            \
            throw std::runtime_error(std::string("failed debug assertion \"") + #EXPR + "\""); \
        }                                                                                      \
    } while (0)
#else
#define RPY_DBG_ASSERT(ARG) assert(ARG)
#endif
#else
#define RPY_DBG_ASSERT(ARG) (void)0
#endif

#define RPY_FALLTHROUGH (void) 0


#endif//ROUGHPY_CORE_MACROS_H
