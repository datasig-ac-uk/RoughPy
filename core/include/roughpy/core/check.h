//
// Created by sam on 13/11/24.
//

#ifndef ROUGHPY_CORE_CHECK_H
#define ROUGHPY_CORE_CHECK_H

#include <stdexcept>
#include <string>
#include <string_view>

#include "check_helpers.h"
#include "macros.h"
#include "string_utils.h"

#if defined(RPY_COMPILER_GCC)
#  define RPY_FUNC_NAME __PRETTY_FUNCTION__
#elif defined(RPY_COMPILER_CLANG)
#  define RPY_FUNC_NAME __builtin_FUNCTION()
#elif defined(RPY_COMPILER_MSVC)
#  define RPY_FUNC_NAME __FUNCTION__
#else
#  define RPY_FUNC_NAME static_cast<const char*>(0)
#endif

namespace rpy::errors {

template <typename E = std::runtime_error>
RPY_NO_RETURN void throw_exception(
        std::string_view user_msg,
        const char* filename,
        int lineno,
        const char* func
)
{
    throw E(string_cat(
            "Error occurred in ",
            filename,
            " at line ",
            lineno,
            '(',
            func,
            "):\n",
            user_msg
    ));
    //    throw E(format_error_message(msg, filename, lineno, func,
    //    boost::stacktrace::stacktrace()));
}

template <typename E = std::runtime_error>
RPY_NO_RETURN void throw_exception(
        const char* user_msg,
        const char* filename,
        int lineno,
        const char* func
)
{
    throw E(string_cat(
            "Error occurred in ",
            filename,
            " at line ",
            lineno,
            '(',
            func,
            "):\n",
            user_msg
    ));
}

}// namespace rpy::errors

#define RPY_THROW_2(EXC_TYPE, MSG)                                             \
    ::rpy::errors::throw_exception<EXC_TYPE>(                                  \
            MSG,                                                               \
            __FILE__,                                                          \
            __LINE__,                                                          \
            RPY_FUNC_NAME                                                      \
    )
#define RPY_THROW_1(MSG) RPY_THROW_2(std::runtime_error, MSG)

#define RPY_THROW_SEL(NUM) RPY_JOIN(RPY_THROW_, NUM)
#define RPY_THROW_CNT_IMPL(_1, _2, COUNT, ...) COUNT
#define RPY_THROW_CNT(...) RPY_THROW_CNT_IMPL(__VA_ARGS__, 2, 1, 0)
#define RPY_THROW(...)                                                         \
    RPY_INVOKE_VA(RPY_THROW_SEL(RPY_COUNT_ARGS(__VA_ARGS__)), (__VA_ARGS__))

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

// Dispatch the check macro on the number of arguments
// See: https://stackoverflow.com/a/16683147/9225581
#define RPY_CHECK_3(EXPR, MSG, TYPE)                                           \
    do {                                                                       \
        if (RPY_UNLIKELY(!(EXPR))) {                                           \
            ::rpy::errors::throw_exception<TYPE>(                              \
                    MSG,                                                       \
                    RPY_FILE_NAME,                                             \
                    __LINE__,                                                  \
                    RPY_FUNC_NAME                                              \
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

// TODO: Ideally have customisable messages for the these checks

#define RPY_CHECK_EQ(a, b, ...)                                                \
    RPY_CHECK(                                                                 \
            (::rpy::compare_equal((a), (b))),                                  \
            "failed check \"" RPY_STRINGIFY((a) == (b)) "\"", __VA_ARGS__)
#define RPY_CHECK_NE(a, b, ...)                                                \
    RPY_CHECK(                                                                 \
            (::rpy::compare_not_equal((a), (b))),                              \
            "failed check \"" RPY_STRINGIFY((a) != (b)) "\"", __VA_ARGS__)
#define RPY_CHECK_LT(a, b, ...)                                                \
    RPY_CHECK(                                                                 \
            (::rpy::compare_less((a), (b))),                                   \
            "failed check \"" RPY_STRINGIFY((a) < (b)) "\"", __VA_ARGS__)
#define RPY_CHECK_LE(a, b, ...)                                                \
    RPY_CHECK(                                                                 \
            (::rpy::compare_less_equal((a), (b))),                             \
            "failed check \"" RPY_STRINGIFY((a) <= (b)) "\"", __VA_ARGS__)
#define RPY_CHECK_GT(a, b, ...)                                                \
    RPY_CHECK(                                                                 \
            (::rpy::compare_greater((a), (b))),                                \
            "failed check \"" RPY_STRINGIFY((a) > (b)) "\"", __VA_ARGS__       \
            )
#define RPY_CHECK_GE(a, b, ...)                                                \
    RPY_CHECK(                                                                 \
            (::rpy::compare_greater_equal((a), (b))),                          \
            "failed check \"" RPY_STRINGIFY((a) >= (b)) "\"", __VA_ARGS__)

#endif// ROUGHPY_CORE_CHECK_H
