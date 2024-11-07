//
// Created by user on 25/01/23.
//

#include "libalgebra_lite/unpacked_tensor_word.h"

#include <algorithm>

namespace lal {

unpacked_tensor_word::unpacked_tensor_word(deg_t width, tensor_basis::key_type key)
    : m_data(), m_width(width)
{
    auto deg = static_cast<deg_t>(key.degree());
    m_data.reserve(deg);
    auto index = key.index();
    for (deg_t i = 0; i<deg; ++i) {
        auto old = index;
        index /= width;
        m_data.emplace_back(old-index*width);
    }
}

template <typename V>
static bool check_letters(deg_t width, const V& letters) {
    using letter_type = typename V::value_type;
    return std::all_of(letters.begin(), letters.end(),
            [width](letter_type i) { return i < width; });
//    for (const auto& item : letters) {
//        if (item >= width) {
//            return false;
//        }
//    }
//    return true;
}

unpacked_tensor_word::unpacked_tensor_word(deg_t width, const std::vector<letter_type>& data)
    : m_data(data.begin(), data.end()), m_width(width)
{
    assert(check_letters(width, m_data));
}
unpacked_tensor_word::unpacked_tensor_word(deg_t width, unpacked_tensor_word::vec_type&& data) noexcept
    : m_data(std::move(data)), m_width(width)
{
    assert(check_letters(width, m_data));
}

unpacked_tensor_word::unpacked_tensor_word(deg_t width, deg_t depth)
    : m_data(depth), m_width(width)
{
}

void unpacked_tensor_word::advance(deg_t number)
{
    const auto degree = m_data.size();
    if (degree == 0) { return; }
    assert(number < std::numeric_limits<letter_type>::max());
    dimn_t position = 0;
    do {
        auto& let = m_data[position];
        let += number;
        number = 0;
        if (let >= m_width) {
            number = let / m_width;
            let = let - number*m_width;
            ++position;
        }
    }
    while (number > 0 && position < degree);
}


unpacked_tensor_word& unpacked_tensor_word::operator++()
{
    advance(1);
    return *this;
}

const unpacked_tensor_word unpacked_tensor_word::operator++(int)
{
   unpacked_tensor_word tmp(*this);
   advance(1);
   return tmp;
}

unpacked_tensor_word::index_type unpacked_tensor_word::pack_with_base(deg_t base, letter_type offset) const noexcept
{
    assert(base >= m_width);
    index_type result = 0;
    auto degree = m_data.size();
    for (dimn_t i=1; i<=degree; ++i) {
        result *= base;
        result += offset + m_data[degree - i];
    }
    return result;
}

unpacked_tensor_word::index_type unpacked_tensor_word::to_index() const noexcept
{
    return pack_with_base(m_width);
}

unpacked_tensor_word::index_type unpacked_tensor_word::to_reverse_index() const noexcept
{
    index_type result = 0;
    for (const auto& let : m_data) {
        result *= m_width;
        result += let;
    }
    return result;
}

typename tensor_basis::key_type unpacked_tensor_word::pack() const noexcept
{
    return typename tensor_basis::key_type{degree(), to_index()};
}

unpacked_tensor_word& unpacked_tensor_word::reverse() noexcept
{
    std::reverse(m_data.begin(), m_data.end());
    return *this;
}

unpacked_tensor_word unpacked_tensor_word::split_left(deg_t left_degree)
{
    if (left_degree >= degree()) {
        unpacked_tensor_word tmp(*this);
        m_data.clear();
        return tmp;
    }

    auto right_size = m_data.size() - left_degree;
    vec_type left_letters(m_data.end()-left_degree, m_data.end());
    m_data.resize(right_size);
    assert(left_letters.size() + m_data.size() == right_size + left_degree);
    return {m_width, std::move(left_letters)};
}

std::pair<unpacked_tensor_word, unpacked_tensor_word> unpacked_tensor_word::split(deg_t left_letters) const
{
    auto right = *this;
    auto left = right.split_left(left_letters);
    return {left, right};
}

unpacked_tensor_word unpacked_tensor_word::operator*(const unpacked_tensor_word &other) const {
    vec_type new_data;
    new_data.reserve(m_data.size() + other.m_data.size());
    new_data.insert(new_data.end(), other.m_data.begin(), other.m_data.end());
    new_data.insert(new_data.end(), m_data.begin(), m_data.end());
    return {std::max(m_width, other.m_width), std::move(new_data)};
}

bool unpacked_tensor_word::operator==(const unpacked_tensor_word& other) const noexcept
{
    return degree() == other.degree() && std::equal(m_data.begin(), m_data.end(), other.m_data.begin());
}
bool unpacked_tensor_word::operator!=(const unpacked_tensor_word& other) const noexcept
{
    return !operator==(other);
}
bool unpacked_tensor_word::operator<(const unpacked_tensor_word& other) const noexcept
{
    auto ldegree = degree();
    auto rdegree = other.degree();
    if (ldegree > rdegree) {
        return false;
    }

    if (ldegree < rdegree) {
        return true;
    }

    return std::lexicographical_compare(m_data.begin(), m_data.end(),
            other.m_data.begin(), other.m_data.end());
}
bool unpacked_tensor_word::operator<=(const unpacked_tensor_word& other) const noexcept
{
    return !operator>(other);
}
bool unpacked_tensor_word::operator>(const unpacked_tensor_word& other) const noexcept
{
    return other.operator<(*this);
}
bool unpacked_tensor_word::operator>=(const unpacked_tensor_word& other) const noexcept
{
    return !operator<(other);
}


} // lal


using namespace lal;

std::ostream& lal::operator<<(std::ostream& os, const unpacked_tensor_word& word)
{
    bool first = true;
    for (deg_t i = 0; i<word.degree(); ++i) {
        if (first) {
            first = false;
        }
        else {
            os << ',';
        }
        os << 1 + word[i];
    }
    return os;
}
