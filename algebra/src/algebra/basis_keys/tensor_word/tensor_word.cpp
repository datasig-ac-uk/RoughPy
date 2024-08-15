//
// Created by sam on 16/02/24.
//

#include "tensor_word.h"

#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::algebra;

TensorWord::TensorWord(const TensorWord& left, const TensorWord& right)
{
    reserve(left.size() + right.size());
    auto pos = insert(end(), left.begin(), left.end());
    insert(pos, right.begin(), right.end());
}

optional<TensorWord> TensorWord::left_parent() const noexcept
{
    if (empty()) { return TensorWord{}; }

    return TensorWord{*begin()};
}

optional<TensorWord> TensorWord::right_parent() const noexcept
{
    if (empty()) { return TensorWord{}; }
    TensorWord result;
    result.assign(++begin(), end());
    return result;
}

void TensorWord::print(std::ostream& os) const
{
    auto it = begin();
    const auto it_end = end();
    if (it != it_end) {
        os << *it;
        ++it;
        for (; it != it_end; ++it) { os << ',' << *it; }
    }
}
deg_t TensorWord::min_alphabet_size() const noexcept
{
    if (empty()) { return 0; }
    return *ranges::max_element(*this);
}

hash_t algebra::hash_value(const TensorWord& tensor_word)
{
    hash_t result = 0;
    Hash<let_t> hasher;
    for (const auto& letter : tensor_word) {
        hash_combine(result, hasher(letter));
    }
    return result;
}
