//
// Created by user on 25/01/23.
//

#ifndef LIBALGEBRA_LITE_UNPACKED_TENSOR_WORD_H
#define LIBALGEBRA_LITE_UNPACKED_TENSOR_WORD_H

#include "implementation_types.h"


#include <boost/container/small_vector.hpp>

#include "tensor_basis.h"

namespace lal {

class LIBALGEBRA_LITE_EXPORT unpacked_tensor_word {
    using letter_type = unsigned short;
    using vec_type = boost::container::small_vector<letter_type, 1>;

    using index_type = typename tensor_basis::key_type::index_type;

    vec_type m_data;
    deg_t m_width;

public:

    template <typename I>
    unpacked_tensor_word(deg_t width, std::initializer_list<I> args)
        : m_data(), m_width(width)
    {
        m_data.reserve(args.size());
        for (auto arg : args) {
            assert(letter_type(arg) < letter_type(m_width));
            m_data.emplace_back(arg);
        }
        std::reverse(m_data.begin(), m_data.end());
    }

    unpacked_tensor_word(deg_t width, const std::vector<letter_type>& data);
    unpacked_tensor_word(deg_t width, vec_type&& data) noexcept;
    unpacked_tensor_word(deg_t width, typename tensor_basis::key_type key);
    explicit unpacked_tensor_word(deg_t width, deg_t depth=0);

public:

    inline deg_t degree() const noexcept { return static_cast<deg_t>(m_data.size()); }
    inline deg_t width() const noexcept { return m_width; }

    template <typename I>
    letter_type operator[](I index) const noexcept {
        assert(dimn_t(index) < m_data.size());
        return m_data[m_data.size() - 1 - index];
    }


private:

    void advance(deg_t number);

public:

    unpacked_tensor_word& operator++();
    const unpacked_tensor_word operator++(int);

    template <typename I>
    unpacked_tensor_word& operator+=(I number) {
        assert(number > 0);
        advance(deg_t(number));
        return *this;
    }

    index_type pack_with_base(deg_t base, letter_type offset=0) const noexcept;
    index_type to_index() const noexcept;
    index_type to_reverse_index() const noexcept;

    typename tensor_basis::key_type pack() const noexcept;


    unpacked_tensor_word& reverse() noexcept;

    unpacked_tensor_word split_left(deg_t left_degree);
    std::pair<unpacked_tensor_word, unpacked_tensor_word> split(deg_t left_letters) const;

    unpacked_tensor_word operator*(const unpacked_tensor_word& other) const;

    bool operator==(const unpacked_tensor_word& other) const noexcept;
    bool operator!=(const unpacked_tensor_word& other) const noexcept;
    bool operator<(const unpacked_tensor_word& other) const noexcept;
    bool operator<=(const unpacked_tensor_word& other) const noexcept;
    bool operator>(const unpacked_tensor_word& other) const noexcept;
    bool operator>=(const unpacked_tensor_word& other) const noexcept;

};

LIBALGEBRA_LITE_EXPORT std::ostream& operator<<(std::ostream& os, const unpacked_tensor_word& word);

} // lal

#endif //LIBALGEBRA_LITE_UNPACKED_TENSOR_WORD_H
