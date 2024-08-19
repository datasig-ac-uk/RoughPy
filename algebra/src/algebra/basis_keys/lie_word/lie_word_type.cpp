//
// Created by sam on 8/13/24.
//

#include "lie_word_type.h"

#include <roughpy/devices/device_handle.h>

#include "lie_word.h"

using namespace rpy;
using namespace rpy::algebra;

LieWordType::LieWordType()
    : Type("lie_word",
           "LieWord",
           {devices::TypeCode::KeyType, sizeof(LieWord), alignof(LieWord), 1},
           devices::traits_of<LieWord>())
{

    {
        auto support = update_support(*this);
        auto& cmps = support->comparison;

        cmps.equals = +[](const void* lptr, const void* rptr) {
            const LieWord& left = *static_cast<const LieWord*>(lptr);
            const LieWord& right = *static_cast<const LieWord*>(rptr);
            return left == right;
        };

        cmps.not_equals = +[](const void* lptr, const void* rptr) {
            const LieWord& left = *static_cast<const LieWord*>(lptr);
            const LieWord& right = *static_cast<const LieWord*>(rptr);
            return left != right;
        };
    }

    set_hash_fn(+[](const void* val) {
        return hash_value(*static_cast<const LieWord*>(val));
    });
}

void* LieWordType::allocate_single() const { return new LieWord{}; }
void LieWordType::free_single(void* ptr) const
{
    delete static_cast<LieWord*>(ptr);
}
bool LieWordType::supports_device(const devices::Device& device) const noexcept
{
    return device->is_host();
}

devices::TypeComparison LieWordType::compare_with(const Type& other
) const noexcept
{
    return Type::compare_with(other);
}
void LieWordType::copy(void* dst, const void* src, dimn_t count) const
{
    const auto* src_ptr = static_cast<const LieWord*>(src);
    auto* dst_ptr = static_cast<LieWord*>(dst);

    ranges::copy_n(src_ptr, count, dst_ptr);
}
void LieWordType::move(void* dst, void* src, dimn_t count) const
{
    auto* src_ptr = static_cast<LieWord*>(src);
    auto* dst_ptr = static_cast<LieWord*>(dst);

    for (dimn_t i = 0; i < count; ++i) { dst_ptr[i] = std::move(src_ptr[i]); }
}
void LieWordType::display(std::ostream& os, const void* ptr) const
{
    const auto& word = *static_cast<const LieWord*>(ptr);
    word.print(os);
}
devices::TypePtr LieWordType::get()
{
    static devices::TypePtr tp(new LieWordType);
    return tp;
}
