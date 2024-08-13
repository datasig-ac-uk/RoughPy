//
// Created by sam on 16/02/24.
//

#include "tensor_word.h"

#include <algorithm>

using namespace rpy;
using namespace rpy::algebra;

constexpr string_view TensorWord::key_name;

string_view TensorWord::key_type() const noexcept { return key_name; }

deg_t TensorWord::min_width() const noexcept
{
    if (m_letters.empty()) { return 0; }
    return static_cast<deg_t>(
            *std::max_element(m_letters.begin(), m_letters.end())
    );
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
