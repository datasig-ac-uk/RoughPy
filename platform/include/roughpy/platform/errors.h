//
// Created by sam on 10/01/24.
//

#ifndef ROUGHPY_PLATFORM_ERRORS_H
#define ROUGHPY_PLATFORM_ERRORS_H

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "roughpy_platform_export.h"
#include <boost/stacktrace.hpp>

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

ROUGHPY_PLATFORM_EXPORT
string format_error_message(string_view user_message, const char* filename, int lineno, const char* func, const boost::stacktrace::stacktrace& st);

template <typename E>
RPY_NO_RETURN RPY_INLINE_ALWAYS void throw_exception(
        const string& msg, const char* filename, int lineno, const char* func
)
{
    throw E(format_error_message(msg, filename, lineno, func, boost::stacktrace::stacktrace()));
}

template <typename E>
RPY_NO_RETURN RPY_INLINE_ALWAYS void throw_exception(
        const char* msg, const char* filename, int lineno, const char* func
)
{
    throw E(format_error_message(msg, filename, lineno, func, boost::stacktrace::stacktrace()));
}

}// namespace errors
}// namespace rpy

#ifndef __CLION_IDE__
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

#else

#  define RPY_CHECK(...)
#  define RPY_THROW(...)

#endif

#endif// ROUGHPY_PLATFORM_ERRORS_H
