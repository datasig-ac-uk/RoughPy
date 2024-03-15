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
