//
// Created by sam on 16/02/24.
//

#ifndef ROUGHPY_TENSOR_WORD_H
#define ROUGHPY_TENSOR_WORD_H

#include <boost/container/small_vector.hpp>

#include <roughpy/core/container/vector.h>
#include <roughpy/platform/alloc.h>

namespace rpy {
namespace algebra {

class TensorWord : public platform::SmallObjectBase
{
    using container_t = containers::SmallVec<let_t, 1>;

    container_t m_letters{};

public:
    static constexpr string_view key_name = "tensor_word";

    using iterator = typename container_t::iterator;
    using const_iterator = typename container_t::const_iterator;
    using reverse_iterator = typename container_t::reverse_iterator;
    using const_reverse_iterator = typename container_t::const_reverse_iterator;

    explicit TensorWord(dimn_t capacity) : m_letters()
    {
        m_letters.reserve(capacity);
    }

    void push_back(let_t letter) { m_letters.push_back(letter); }

    string_view key_type() const noexcept;

    dimn_t degree() const noexcept { return m_letters.size(); }

    auto begin() noexcept -> decltype(m_letters.begin())
    {
        return m_letters.begin();
    }
    auto end() noexcept -> decltype(m_letters.end()) { return m_letters.end(); }

    auto begin() const noexcept -> decltype(m_letters.begin())
    {
        return m_letters.begin();
    }
    auto end() const noexcept -> decltype(m_letters.end())
    {
        return m_letters.end();
    }

    auto cbegin() const noexcept -> decltype(m_letters.cbegin())
    {
        return m_letters.cbegin();
    }
    auto cend() const noexcept -> decltype(m_letters.cend())
    {
        return m_letters.cend();
    }

    auto rbegin() const noexcept -> decltype(m_letters.rbegin())
    {
        return m_letters.rbegin();
    }
    auto rend() const noexcept -> decltype(m_letters.rbegin())
    {
        return m_letters.rbegin();
    }

    auto rbegin() noexcept -> decltype(m_letters.rbegin())
    {
        return m_letters.rbegin();
    }
    auto rend() noexcept -> decltype(m_letters.rbegin())
    {
        return m_letters.rbegin();
    }

    RPY_NO_DISCARD deg_t min_width() const noexcept;
};

hash_t hash_value(const TensorWord& tensor_word);

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_TENSOR_WORD_H
