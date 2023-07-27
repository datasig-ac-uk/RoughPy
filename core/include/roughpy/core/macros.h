// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 04/03/23.
//

#ifndef ROUGHPY_CORE_MACROS_H
#define ROUGHPY_CORE_MACROS_H

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef __has_builtin
#  define RPY_HAS_BUILTIN(x) __has_builtin(x)
#else
#  define RPY_HAS_BUILTIN(x) 0
#endif

#ifdef __has_feature
#  define RPY_HAS_FEATURE(FEAT) __has_feature(FEAT)
#else
#  define RPY_HAS_FEATURE(FEAT) 0
#endif

#ifdef __has_cpp_attribute
#  define RPY_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#  define RPY_HAS_CPP_ATTRIBUTE(x) 0
#endif

#ifdef __has_include
#  define RPY_HAS_INCLUDE(x) __has_include(x)
#else
#  define RPY_HAS_INCLUDE(x) 0
#endif

/*
 * MSVC has a bug where it treats __VA_ARGS__ as a single token in argument
 * lists. To combat this, we use RPY_INVOKE_VA to invoke the overload macro with
 * the __VA_ARGS__ in the token stream so that proper expansion occurs.
 * See: https://stackoverflow.com/a/9338429
 */
#define RPY_INVOKE_VA(X, Y) X Y

#define RPY_COUNT_ARGS_1(_4, _3, _2, _1, count, ...) count
#define RPY_COUNT_ARGS(...)                                                    \
    RPY_INVOKE_VA(RPY_COUNT_ARGS_1, (__VA_ARGS__, 4, 3, 2, 1, 0))

#define RPY_STRINGIFY_IMPL(ARG) #ARG
#define RPY_STRINGIFY(ARG) RPY_STRINGIFY_IMPL(ARG)

#define RPY_JOIN_IMPL_IMPL(LHS, RHS) LHS##RHS
#define RPY_JOIN_IMPL(LHS, RHS) RPY_JOIN_IMPL_IMPL(LHS, RHS)
#define RPY_JOIN(LHS, RHS) RPY_JOIN_IMPL(LHS, RHS)

#define RPY_IDENTITY(ARG) ARG

#if (defined(_DEBUG) || !defined(NDEBUG) || !defined(__OPTIMIZE__))            \
        && !defined(RPY_DEBUG)
#  define RPY_DEBUG
#else
#  undef RPY_DEBUG
#endif

#if defined(_MSC_VER) && defined(_MSVC_LANG)
#  define RPY_CPP_VERSION _MSVC_LANG
#  define RPY_MSVC _MSC_VER
#else
#  define RPY_CPP_VERSION __cplusplus
#  undef RPY_MSVC
#endif

#if defined(__GNUC__) && !defined(__clang__)
#  define RPY_GCC __GNUC__
#else
#  undef RPY_GCC
#endif

#ifdef __clang__
#  define RPY_CLANG __clang__
#else
#  undef RPY_CLANG
#endif

#if defined(__linux__)
#  define RPY_PLATFORM_LINUX 1
#elif defined(__APPLE__) && defined(__MACH__)
#  define RPY_PLATFORM_MACOS 1
#elif defined(_WIN32) || defined(WIN32)
#  define RPY_PLATFORM_WINDOWS 1
#endif

#if defined(RPY_PLATFORM_WINDOWS) || defined(__CYGWIN__)
#  define RPY_COMPILING_DLL
#endif

#ifndef RPY_DISABLE_EXPORTS
#  ifdef RPY_COMPILING_DLL
#    if RPY_BUILDING_LIBRARY
#      if defined(RPY_MSVC)
#        define RPY_EXPORT __declspec(dllexport)
#        define RPY_LOCAL
#      else
#        define RPY_EXPORT __attribute__((dllexport))
#        define RPY_LOCAL
#      endif
#    else
#      if defined(RPY_MSVC)
#        define RPY_EXPORT __declspec(dllimport)
#        define RPY_LOCAL
#      else
#        define RPY_EXPORT __attribute__((dllimport))
#      endif
#    endif
#  else
#    if (defined(RPY_GCC) && RPY_GCC >= 4) || defined(RPY_CLANG)
#      if defined(RPY_BUILDING_LIBRARY)
#        define RPY_EXPORT __attribute__((visibility("default")))
#        define RPY_LOCAL __attribute__((visibility("hidden")))
#      else
#        define RPY_EXPORT __attribute__((visibility("default")))
#        define RPY_LOCAL __attribute__((visibility("hidden")))
#      endif
#    else
#      define RPY_EXPORT
#      define RPY_LOCAL
#    endif
#  endif
#else
#  define RPY_EXPORT
#  define RPY_LOCAL
#endif

#if RPY_CPP_VERSION >= 201403L
#  define RPY_CPP_14
#else
#  undef RPY_CPP_14
#endif

#if RPY_CPP_VERSION >= 201703L
#  define RPY_CPP_17
#else
#  undef RPY_CPP_17
#endif

#ifdef RPY_CPP_17
#  define RPY_UNUSED [[maybe_unused]]
#  define RPY_NO_DISCARD [[nodiscard]]
#  define RPY_NO_RETURN [[noreturn]]
#  define RPY_IF_CONSTEXPR constexpr
#else
#  define RPY_UNUSED
#  define RPY_NO_DISCARD
#  define RPY_IF_CONSTEXPR
#  define RPY_NO_RETURN
#endif

#if defined(RPY_GCC) || defined(RPY_CLANG)
#  define RPY_UNUSED_VAR __attribute__((unused))
#else
#  define RPY_UNUSED_VAR
#endif

#if defined(RPY_GCC) || defined(RPY_CLANG)
#  define RPY_UNREACHABLE() (__builtin_unreachable())
#  define RPY_UNREACHABLE_RETURN(...) RPY_UNREACHABLE()
#elif defined(RPY_MSVC)
#  define RPY_UNREACHABLE() (__assume(false))
#  define RPY_UNREACHABLE_RETURN(...)                                          \
      RPY_UNREACHABLE();                                                       \
      return __VA_ARGS__
#else
#  define RPY_UNREACHABLE() abort()
#  define RPY_UNREACHABLE_RETURN(...)                                          \
      RPY_UNREACHABLE();                                                       \
      return __VA_ARGS__
#endif

#ifdef RPY_MSVC
#  define RPY_PRAGMA(ARG) __pragma(ARG)
#else
#  define RPY_PRAGMA(ARG) _Pragma(RPY_STRINGIFY(ARG))
#endif

// Macros that control optimisations

#if defined(__OPTIMIZE__) || !defined(RPY_DEBUG)
#  if defined(_WIN32) || defined(_WIN64)
#    define RPY_INLINE_ALWAYS __forceinline
#  elif defined(__GNUC__) || defined(__clang__)
#    define RPY_INLINE_ALWAYS inline __attribute__((always_inline))
#  else
#    define RPY_INLINE_ALWAYS inline
#  endif
#else
#  define RPY_INLINE_ALWAYS inline
#endif

#ifdef RPY_MSVC
#  define RPY_INLINE_NEVER __declspec(noinline)
#elif defined(RPY_GCC) || defined(RPY_CLANG)
#  define RPY_INLINE_NEVER __attribute__((never_inline))
#else
#  define RPY_INLINE_NEVER
#endif

#if defined(RPY_GCC) || defined(RPY_CLANG)
#  define RPY_RESTRICT __restrict__
#elif defined(RPY_MSVC)
#  define RPY_RESTRICT __restrict
#else
#  define RPY_RESTRICT
#endif

#if defined(RPY_GCC) || defined(RPY_CLANG)
#  define RPY_LIKELY(COND) (__builtin_expect(static_cast<bool>(COND), 1))
#  define RPY_UNLIKELY(COND) (__builtin_expect(static_cast<bool>(COND), 0))
#else
#  define RPY_LIKELY(COND) (COND)
#  define RPY_UNLIKELY(COND) (COND)
#endif

#define RPY_FALLTHROUGH (void) 0

#if defined(RPY_PLATFORM_WINDOWS)
#  if RPY_BUILDING_LIBRARY
#    define RPY_TEMPLATE_EXTERN
#    define RPY_EXPORT_TEMPLATE
#    define RPY_EXPORT_INSTANTIATION RPY_EXPORT
#  else
#    define RPY_TEMPLATE_EXTERN
#    define RPY_EXPORT_TEMPLATE RPY_EXPORT
#    define RPY_EXPORT_INSTANTIATION
#  endif
#else
#  define RPY_TEMPLATE_EXTERN extern
#  define RPY_EXPORT_TEMPLATE RPY_EXPORT
#  define RPY_EXPORT_INSTANTIATION
#endif

// Sanitizer supports
#ifdef RPY_CLANG
#  define RPY_NO_UBSAN __attribute__((no_sanitize("undefined")))
#else
#  define RPY_NO_UBSAN
#endif

#if RPY_HAS_FEATURE(address_sanitizer)
#  define RPY_NO_ASAN __attribute__((no_sanitize("address")))
#else
#  define RPY_NO_ASAN
#endif

// Warning and error control
#if defined(RPY_MSVC)
#  define RPY_WARNING_PUSH RPY_PRAGMA(warning(push))
#  define RPY_WARNING_POP RPY_PRAGMA(warning(pop))
#elif defined(RPY_GCC)
#  define RPY_WARNING_PUSH RPY_PRAGMA(GCC diagnostic push)
#  define RPY_WARNING_POP RPY_PRAGMA(GCC diagnostic pop)
#elif defined(RPY_CLANG)
#  define RPY_WARNING_PUSH RPY_PRAGMA(clang diagnostic push)
#  define RPY_WARNING_POP RPY_PRAGMA(clang diagnostic pop)
#else
#  define RPY_WARNING_PUSH
#  define RPY_WARNING_POP
#endif

#ifdef RPY_MSVC
#  define RPY_MSVC_DISABLE_WARNING(ARG) RPY_PRAGMA(warning(disable : ARG))
#else
#  define RPY_MSVC_DISABLE_WARNING(ARG)
#endif

#ifdef RPY_GCC
#  define RPY_GCC_DISABLE_WARNING(ARG)                                         \
      RPY_PRAGMA(GCC diagnostic ignored RPY_STRINGIFY(ARG))
#else
#  define RPY_GCC_DISABLE_WARNING(ARG)
#endif

#ifdef RPY_CLANG
#  define RPY_CLANG_DISABLE_WARNING(ARG)                                       \
      RPY_PRAGMA(clang diagnostic ignored RPY_STRINGIFY(ARG))
#else
#  define RPY_CLANG_DISABLE_WARNING(ARG)
#endif

#if defined(RPY_GCC)
#  define RPY_FUNC_NAME __PRETTY_FUNCTION__
#elif defined(RPY_CLANG)
#  define RPY_FUNC_NAME __builtin_FUNCTION()
#elif defined(RPY_MSVC)
#  define RPY_FUNC_NAME __FUNCTION__
#else
#  define RPY_FUNC_NAME static_cast<const char*>(0)
#endif

#if defined(RPY_GCC)
#  define RPY_FILE_NAME __FILE__
#elif defined(RPY_CLANG)
#  define RPY_FILE_NAME __FILE__
#elif defined(RPY_MSVC)
#  define RPY_FILE_NAME __FILE__
#else
#  define RPY_FILE_NAME __FILE__
#endif

/*
 * Check macro definition.
 *
 * This macro checks that the given expression evaluates to true (under the
 * assumption that it will usually be true), and throws an error if this
 * evaluates to false.
 *
 * Optionally, one can provide a message string literal that will be used
 * instead of the default, and an optional error type. The default error type
 * is a std::runtime_error.
 */
namespace rpy {
namespace errors {

template <typename E>
RPY_NO_RETURN RPY_INLINE_ALWAYS void throw_exception(
        std::string msg, const char* filename, int lineno, const char* func
)
{
    std::stringstream ss;
    ss << msg << " at lineno " << lineno << " in " << filename
       << " in function " << func;
    throw E(ss.str());
}

template <typename E>
RPY_NO_RETURN RPY_INLINE_ALWAYS void throw_exception(
        const char* msg, const char* filename, int lineno, const char* func
)
{
    std::stringstream ss;
    ss << msg << " at lineno " << lineno << " in " << filename
       << " in function " << func;
    throw E(ss.str());
}

}// namespace errors
}// namespace rpy

// Dispatch the check macro on the number of arguments
// See: https://stackoverflow.com/a/16683147/9225581
#define RPY_CHECK_3(EXPR, MSG, TYPE)                                           \
    do {                                                                       \
        if (RPY_UNLIKELY(!(EXPR))) {                                           \
            ::rpy::errors::throw_exception<TYPE>(                              \
                    MSG, RPY_FILE_NAME, __LINE__, RPY_FUNC_NAME                \
            );                                                                 \
        }                                                                      \
    } while (0)

#define RPY_CHECK_2(EXPR, MSG) RPY_CHECK_3(EXPR, MSG, std::runtime_error)

#define RPY_CHECK_1(EXPR) RPY_CHECK_2(EXPR, "failed check \"" #EXPR "\"")

#define RPY_CHECK_CNT_IMPL(_1, _2, _3, COUNT, ...) COUNT
// Always pass one more argument than expected, so clang doesn't complain about
// empty parameter packs.
#define RPY_CHECK_CNT(...) RPY_CHECK_CNT_IMPL(__VA_ARGS__, 3, 2, 1, 0)
#define RPY_CHECK_SEL(NUM) RPY_JOIN(RPY_CHECK_, NUM)

#define RPY_CHECK(...)                                                         \
    RPY_INVOKE_VA(RPY_CHECK_SEL(RPY_COUNT_ARGS(__VA_ARGS__)), (__VA_ARGS__))

#define RPY_THROW_2(EXC_TYPE, MSG)                                             \
    ::rpy::errors::throw_exception<EXC_TYPE>(                                  \
            MSG, __FILE__, __LINE__, RPY_FUNC_NAME                             \
    )
#define RPY_THROW_1(MSG) RPY_THROW_2(std::runtime_error, MSG)

#define RPY_THROW_SEL(NUM) RPY_JOIN(RPY_THROW_, NUM)
#define RPY_THROW_CNT_IMPL(_1, _2, COUNT, ...) COUNT
#define RPY_THROW_CNT(...) RPY_THROW_CNT_IMPL(__VA_ARGS__, 2, 1, 0)
#define RPY_THROW(...)                                                         \
    RPY_INVOKE_VA(RPY_THROW_SEL(RPY_COUNT_ARGS(__VA_ARGS__)), (__VA_ARGS__))

#ifdef RPY_DEBUG
#  define RPY_DBG_ASSERT(ARG) assert(ARG)
#else
#  define RPY_DBG_ASSERT(ARG) (void) 0
#endif

#endif// ROUGHPY_CORE_MACROS_H
