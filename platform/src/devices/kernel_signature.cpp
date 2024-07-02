//
// Created by sam on 7/2/24.
//

#include "kernel.h"

#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

dimn_t KernelSignature::add_type_param()
{
    const auto size = m_types.size();
    m_types.emplace_back(nullptr);
    return size;
}

dimn_t KernelSignature::add_type_param(const Type& tp)
{
    const auto begin = m_types.begin();
    const auto end = m_types.end();
    const auto it = ranges::find(begin, end, &tp);
    if (it != end) { return static_cast<dimn_t>(it - begin); }

    const auto size = m_types.size();
    m_types.emplace_back(&tp);
    return size;
}

void KernelSignature::add_param(
        KernelArgumentType kind,
        std::variant<dimn_t, const Type&> type
)
{
    dimn_t type_idx = 0;
    if (holds_alternative<dimn_t>(type)) {
        type_idx = get<dimn_t>(type);
        RPY_CHECK(type_idx < m_types.size());
    } else {
        type_idx = add_type_param(get<const Type&>(type));
    }

    m_params.emplace_back(kind, static_cast<uint8_t>(type_idx));
}
