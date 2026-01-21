//
// Created by sam on 13/11/24.
//

#ifndef ROUGHPY_CORE_STRINGS_H
#define ROUGHPY_CORE_STRINGS_H


#include <string>
#include <string_view>
#include <utility>

namespace rpy {

namespace dtl {


inline void string_append(std::string& result, std::string&& other)
{
    result.append(std::move(other));
}

inline void string_append(std::string& result, std::string_view other)
{
    result.append(other);
}

inline void string_append(std::string& result, const char* other)
{
    result.append(other);
}

inline void string_append(std::string& result, const char& other)
{
    result.push_back(other);
}

template <size_t N>
void string_append(std::string& result, const char (&other)[N])
{
    result.append(other);
}
template <typename T>
void string_append(std::string& result, const T& other)
{
    result.append(std::to_string(other));
}

inline void string_append_all(std::string& result) {}

template <typename Arg, typename... Args>
void string_append_all(std::string& result, Arg&& arg, Args&&... args)
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
std::string string_cat(Args&&... args)
{
    std::string result;
    dtl::string_append_all(result, std::forward<Args>(args)...);
    return result;
}


namespace dtl {

template <typename Joiner, typename Arg>
void join_all(Joiner&& join_fn, Arg&& arg)
{
    join_fn(arg);
}

template <typename Joiner, typename Arg, typename... Args>
void join_all(Joiner&& join_fn, Arg&& arg, Args&&... args)
{
    join_fn(arg);
    join_all(std::forward<Joiner>(join_fn), std::forward<Args>(args)...);
}


}


/**
 * @brief Concatenates multiple strings with a joiner.
 *
 * @param joiner The string used to join the arguments.
 * @param args The strings to be joined.
 * @return The joined string.
 */
template <typename S, typename... Args>
std::string string_join(const S& joiner, Args&&... args)
{
    std::string result;
    bool first = true;
    auto append_arg = [&](const auto& arg) {
        if (!first) { dtl::string_append(result, joiner); }
        dtl::string_append(result, arg);
        first = false;
    };
    dtl::join_all(append_arg, std::forward<Args>(args)...);
    return result;
}

}// namespace rpy





#endif //ROUGHPY_CORE_STRINGS_H
