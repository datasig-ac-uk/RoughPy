//
// Created by sam on 4/26/24.
//

#ifndef ROUGHPY_CORE_ERRORS_H
#define ROUGHPY_CORE_ERRORS_H

#include "macros.h"
#include "types.h"
#include "strings.h"

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

/**
 * @brief Throws an exception with a formatted error message.
 *
 * This method throws an exception with a formatted error message that includes
 * the user-provided message, filename, line number, and function name.
 *
 * @param msg The user-provided message to include in the error message.
 * @param filename The name of the file where the error occurred.
 * @param lineno The line number where the error occurred.
 * @param func The name of the function where the error occurred.
 *
 * @return None.
 */
template <typename E>
RPY_NO_RETURN RPY_INLINE_ALWAYS void throw_exception(
        const string& msg,
        const char* filename,
        int lineno,
        const char* func
)
{
    throw E(msg + " at line " + std::to_string(lineno) + " in file " + filename
            + '(' + func + ')');
}

/**
 * @brief Throws an exception with user-provided message, filename, line number,
 * and function name.
 *
 * This function throws an exception with the provided user message, filename,
 * line number, and function name. It uses the format_error_message function to
 * format the error message string before throwing the exception.
 *
 * @param msg The user-provided message to include in the exception.
 * @param filename The name of the file where the exception is thrown.
 * @param lineno The line number where the exception is thrown.
 * @param func The name of the function where the exception is thrown.
 *
 * @return This function does not return a value.
 *
 * @see format_error_message
 */
template <typename E>
RPY_NO_RETURN RPY_INLINE_ALWAYS void throw_exception(
        const char* msg,
        const char* filename,
        int lineno,
        const char* func
)
{
    throw E(string(msg) + " at line " + std::to_string(lineno) + " in file "
            + filename + '(' + func + ')');
}

}// namespace errors
}// namespace rpy

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

#define RPY_CHECK_1(EXPR)                                                      \
    RPY_CHECK_2(EXPR, "failed check \"" RPY_STRINGIFY(EXPR) "\"")

#define RPY_CHECK_CNT_IMPL(_1, _2, _3, COUNT, ...) COUNT
// Always pass one more argument than expected, so clang doesn't complain about
// empty parameter packs.
#define RPY_CHECK_CNT(...) RPY_CHECK_CNT_IMPL(__VA_ARGS__, 3, 2, 1, 0)
#define RPY_CHECK_SEL(NUM) RPY_JOIN(RPY_CHECK_, NUM)

#define RPY_CHECK(...)                                                         \
    RPY_INVOKE_VA(RPY_CHECK_SEL(RPY_COUNT_ARGS(__VA_ARGS__)), (__VA_ARGS__))

#define RPY_CHECK_EQ(LEFT, RIGHT, ...)                                         \
    RPY_CHECK(((LEFT) == (RIGHT)), __VA_ARGS__)

#define RPY_CHECK_NOTNULL(PTR, ...) RPY_CHECK(((PTR) != nullptr), __VA_ARGS__)
#define RPY_CHECK_ZERO(VAL, ...) RPY_CHECK(((VAL) == 0), __VA_ARGS__)

#define RPY_THROW(EXC_TYPE, ...)                                               \
    ::rpy::errors::throw_exception<EXC_TYPE>(                                  \
            (__VA_ARGS__),                                   \
            __FILE__,                                                          \
            __LINE__,                                                          \
            RPY_FUNC_NAME                                                      \
    )

#endif// ROUGHPY_CORE_ERRORS_H
