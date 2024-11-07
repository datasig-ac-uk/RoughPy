#ifndef LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_DETAIL_MACROS_H_
#define LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_DETAIL_MACROS_H_

#include <cassert>
#include <stdexcept>

#define LAL_STRINGIFY_IMPL(ARG) #ARG
#define LAL_STRINGIFY(ARG) LAL_STRINGIFY_IMPL(ARG)

#define LAL_JOIN_IMPL2(LHS, RHS) LHS##RHS
#define LAL_JOIN_IMPL(LHS, RHS) LAL_JOIN_IMPL2(LHS, RHS)
#define LAL_JOIN(LHS, RHS) LAL_JOIN_IMPL(LHS, RHS)

#if defined(_MSV_VER) && defined(_MSVC_LANG)
#  define LAL_MSVC _MSC_VER
#  define LAL_CPP_VERSION _MSVC_LANG
#else
#  define LAL_CPP_VERSION __cplusplus
#  undef LAL_MSVC
#endif

#if defined(__GNUC__) && !defined(__clang__)
#  define LAL_GCC __GNUC__
#else
#  undef LAL_GCC
#endif

#ifdef __clang__
#  define LAL_CLANG __clang__
#else
#  undef LAL_CLANG
#endif

#if defined(__linux__)
#  define LAL_PLATFORM_LINUX 1
#elif defined(__APPLE__) && defined(__MACH__)
#  define LAL_PLATFORM_OSX 1
#elif defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__)
#  define LAL_PLATFORM_WINDOWS 1
#endif

#ifdef LAL_PLATFORM_WINDOWS
#  define LAL_COMPILE_DLL
#endif

#ifdef __has_builtin
#  define LAL_HAS_BUILTIN(x) __has_builtin(x)
#else
#  define LAL_HAS_BUILTIN(x) 0
#endif

#ifdef __has_feature
#  define LAL_HAS_FEATURE(x) (__has_feature(x))
#else
#  define LAL_HAS_FEATURE(x) 0
#endif

#ifdef __has_cpp_attribute
#  define LAL_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#  define LAL_HAS_CPP_ATTRIBUTE(x) 0
#endif

#ifdef __has_attribute
#  define LAL_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#  define LAL_HAS_ATTRIBUTE(x) 0
#endif

#ifdef __has_extension
#  define LAL_HAS_EXTENSION(x) __has_extension(x)
#else
#  define LAL_HAS_EXTENSION(x) LAL_HAS_FEATURE(x)
#endif

#if defined(_DEBUG) || !defined(NDEBUG)                                        \
        || !defined(__OPTIMIZE__) && !defined(LAL_DEBUG)
#  define LAL_DEBUG 1
#else
#  undef LAL_DEBUG
#endif

#ifdef LAL_MSVC
#  define LAL_PRAGMA(ARG) __pragma(ARG)
#else
#  define LAL_PRAGMA(ARG) _Pragma(LAL_STRINGIFY(ARG))
#endif

#ifndef LAL_DEBUG
#  ifdef LAL_MSVC
#    define LAL_INLINE_ALWAYS __forceinline
#  elif LAL_HAS_ATTRIBUTE(always_inline)
#    define LAL_INLINE_ALWAYS inline __attribute__((always_inline))
#  else
#    define LAL_INLINE_ALWAYS inline
#  endif
#else
#  define LAL_INLINE_ALWAYS inline
#endif

#ifdef LAL_MSVC
#  define LAL_RESTRICT __restrict
#elif defined(LAL_GCC) || defined(LAL_CLANG)
#  define LAL_RESTRICT __restrict__
#else
#  define LAL_RESTRICT
#endif

#if LAL_HAS_CPP_ATTRIBUTE(maybe_unused)
#  define LAL_UNUSED [[maybe_unused]]
#else
#  define LAL_UNUSED
#endif

#if LAL_HAS_CPP_ATTRIBUTE(nodiscard)
#  define LAL_NO_DISCARD [[nodiscard]]
#else
#  define LAL_NO_DISCARD
#endif

#if LAL_HAS_CPP_ATTRIBUTE(noreturn)
#  define LAL_NO_RETURN [[noreturn]]
#else
#  define LAL_NO_RETURN
#endif





#endif// LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_DETAIL_MACROS_H_
