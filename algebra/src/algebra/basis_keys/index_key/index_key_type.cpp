//
// Created by sam on 8/15/24.
//

#include "index_key_type.h"

#include <roughpy/devices/value.h>

using namespace rpy;
using namespace rpy::algebra;

IndexKeyType::IndexKeyType()
    : Type("index_key",
           "IndexKey",
           devices::type_info<dimn_t>(),
           devices::traits_of<dimn_t>())
{

    {
        auto support = update_support(*this);

        auto& cmp = support->comparison;
        cmp.equals = +[](const void* left_ptr, const void* right_ptr) {
            const auto& left = *static_cast<const dimn_t*>(left_ptr);
            const auto& right = *static_cast<const dimn_t*>(right_ptr);
            return left == right;
        };
        cmp.not_equals = +[](const void* left_ptr, const void* right_ptr) {
            const auto& left = *static_cast<const dimn_t*>(left_ptr);
            const auto& right = *static_cast<const dimn_t*>(right_ptr);
            return left != right;
        };

        cmp.less = +[](const void* left_ptr, const void* right_ptr) {
            const auto& left = *static_cast<const dimn_t*>(left_ptr);
            const auto& right = *static_cast<const dimn_t*>(right_ptr);
            return left < right;
        };
        cmp.less_equal = +[](const void* left_ptr, const void* right_ptr) {
            const auto& left = *static_cast<const dimn_t*>(left_ptr);
            const auto& right = *static_cast<const dimn_t*>(right_ptr);
            return left <= right;
        };

        cmp.greater = +[](const void* left_ptr, const void* right_ptr) {
            const auto& left = *static_cast<const dimn_t*>(left_ptr);
            const auto& right = *static_cast<const dimn_t*>(right_ptr);
            return left > right;
        };
        cmp.greater_equal = +[](const void* left_ptr, const void* right_ptr) {
            const auto& left = *static_cast<const dimn_t*>(left_ptr);
            const auto& right = *static_cast<const dimn_t*>(right_ptr);
            return left >= right;
        };
        cmp.is_zero = +[](const void* arg) {
            return *static_cast<const dimn_t*>(arg) == 0;
        };
    }

    set_hash_fn(+[](const void* arg) {
        return *static_cast<const dimn_t*>(arg);
    });
}

void* IndexKeyType::allocate_single() const { return new dimn_t{}; }
void IndexKeyType::free_single(void* ptr) const
{
    delete static_cast<dimn_t*>(ptr);
}
bool IndexKeyType::supports_device(const devices::Device& device) const noexcept
{
    return Type::supports_device(device);
}
void IndexKeyType::copy(void* dst, const void* src, dimn_t count) const
{
    std::memcpy(dst, src, count * sizeof(dimn_t));
}
void IndexKeyType::move(void* dst, void* src, dimn_t count) const
{
    std::memcpy(dst, src, count * sizeof(dimn_t));
}
void IndexKeyType::display(std::ostream& os, const void* ptr) const
{
    os << *static_cast<const dimn_t*>(ptr);
}
devices::ConstReference IndexKeyType::zero() const
{
    static constexpr dimn_t zero = 0;
    return devices::ConstReference{&zero, this};
}
devices::ConstReference IndexKeyType::one() const
{
    static constexpr dimn_t one = 1;
    return devices::ConstReference{&one, this};
}

devices::TypePtr IndexKeyType::get()
{
    static devices::TypePtr type(new IndexKeyType);
    return type;
}
