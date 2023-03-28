//
// Created by user on 04/03/23.
//

#ifndef ROUGHPY_CORE_MACROS_H
#define ROUGHPY_CORE_MACROS_H



#if (defined(_DEBUG) || !defined(NDEBUG) || !defined(__OPTIMIZE__)) && !defined(RPY_DEBUG)
#   define RPY_DEBUG
#endif

#if defined(_MSC_VER) && defined(_MSVC_LANG)
#   define RPY_CPP_VERSION _MSVC_LANG
#else
#   define RPY_CPP_VERSION __cplusplus
#endif


#if defined(__GNUC__) || defined(__clang__)
#   define RPY_UNUSED __attribute__((unused))
#   define RPY_USED __attribute__((used))
#else
#   define RPY_UNUSED
#   define RPY_USED
#endif


#if defined(__GNUC__) || defined(__clang__)
#    define RPY_UNREACHABLE() (__builtin_unreachable())
#    define RPY_UNREACHABLE_RETURN(...) RPY_UNREACHABLE()
#elif defined(_MSC_VER)
#    define RPY_UNREACHABLE() (__assume(false))
#    define RPY_UNREACHABLE_RETURN(...) RPY_UNREACHABLE(); return __VA_ARGS__
#else
#    define RPY_UNREACHABLE()
#    define RPY_UNREACHABLE_RETURN(...) RPY_UNREACHABLE(); return __VA_ARGS__
#endif


// Macros that control optimisations

#if defined(__OPTIMIZE__) || !defined(RPY_DEBUG)
#   if defined(_WIN32) || defined(_WIN64)
#       define RPY_INLINE_ALWAYS __forceinline
#   elif defined(__GNUC__) || defined(__clang__)
#       define RPY_INLINE_ALWAYS inline __attribute__((always_inline))
#   else
#       define RPY_INLINE_ALWAYS inline
#   endif
#endif

#if defined(_WIN32) || defined(_WIN64)
#   define RPY_INLINE_NEVER __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#   define RPY_INLINE_NEVER __attribute__((never_inline))
#else
#   define RPY_INLINE_NEVER
#endif

#if defined(__GNUC__) || defined(__clang__)
#   define RPY_RESTRICT(ARG) ARG __restrict__
#elif defined(_MSC_VER)
#   define RPY_RESTRICT(ARG) ARG __restrict
#endif



#define RPY_STRINGIFY_IMPL(ARG) #ARG
#define RPY_STRINGIFY(ARG) RPY_STRINGIFY_IMPL(ARG)

#define RPY_JOIN_IMPL_IMPL(LHS, RHS) X ## Y
#define RPY_JOIN_IMPL(LHS, RHS) RPY_JOIN_IMPL_IMPL(LHS, RHS)
#define RPY_JOIN(LHS, RHS) RPY_JOIN_IMPL(LHS, RHS)


#endif//ROUGHPY_CORE_MACROS_H
