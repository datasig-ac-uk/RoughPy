#ifndef ROUGHPY_COMPUTE_COMMON_ITERATORS_HPP
#define ROUGHPY_COMPUTE_COMMON_ITERATORS_HPP


#include <iterator>
#include <type_traits>
#include <utility>


namespace rpy::compute {

// Iterator category checks
template <typename I>
inline constexpr bool is_forward_iterator_v = std::is_base_of_v<
std::forward_iterator_tag,
    typename std::iterator_traits<I>::iterator_category>;

template <typename I>
inline constexpr bool is_bidirectional_iterator_v = std::is_base_of_v<
    std::bidirectional_iterator_tag,
    typename std::iterator_traits<I>::iterator_category>;

template <typename I>
inline constexpr bool is_random_access_v = std::is_base_of_v<
    std::random_access_iterator_tag,
    typename std::iterator_traits<I>::iterator_category>;







} // namespace rpy::compute

#endif //ROUGHPY_COMPUTE_COMMON_ITERATORS_HPP