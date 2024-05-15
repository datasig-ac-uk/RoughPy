//
// Created by sam on 5/15/24.
//

#ifndef ROUGHPY_CORE_STRINGS_H
#define ROUGHPY_CORE_STRINGS_H

#include "types.h"

namespace rpy {

namespace dtl {

inline void string_append(string& result, const string& other)
{
    result.append(other);
}

inline void string_append(string& result, string&& other)
{
    result.append(std::move(other));
}

inline void string_append(string& result, const string_view& other)
{
    result.append(other);
}

inline void string_append(string& result, const char& other)
{
    result.push_back(other);
}

template <size_t N>
void string_append(string& result, const char (&other)[N])
{
    result.append(other);
}
template <typename T>
void string_append(string& result, const T& other)
{
    result.append(std::to_string(other));
}

inline void string_append_all(string& result) {}

template <typename Arg, typename... Args>
void string_append_all(string& result, Arg&& arg, Args&&... args)
{
    string_append(result, std::forward<Arg>(arg));
    string_append_all(result, std::forward<Args>(args)...);
}

}// namespace dtl

/**
 * @brief Concatenates multiple strings into a single string.
 *
 * This function takes multiple arguments and concatenates them into a single
 * string, using the + operator to concatenate each argument to the previous
 * one.
 *
 * @param args The arguments to concatenate.
 * @return The concatenated string.
 *
 * @tparam Args The types of the arguments. Must be convertible to string.
 *
 *
 * @see std::string
 */
template <typename... Args>
string string_cat(Args&&... args)
{
    string result;
    dtl::string_append_all(result, std::forward<Args>(args)...);
    return result;
}

}// namespace rpy

#endif// ROUGHPY_CORE_STRINGS_H
