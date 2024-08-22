

//
// Created by sam on 8/12/24.
//

#include "tensor_word_type.h"

#include "roughpy/devices/device_handle.h"
#include "tensor_word.h"

#include <roughpy/core/ranges.h>
#include <roughpy/devices/buffer.h>
#include <roughpy/devices/type.h>
#include <roughpy/devices/value.h>

using namespace rpy;
using namespace rpy::algebra;

TensorWordType::TensorWordType()
    : Type("tensor_word",
           "TensorWord",
           {devices::TypeCode::KeyType,
            sizeof(TensorWord),
            alignof(TensorWord),
            1},
           devices::traits_of<TensorWord>())
{

    {
        auto support = this->update_support(*this);

        support->comparison.equals = +[](const void* vlhs, const void* vrhs) {
            const auto& lhs = *static_cast<const TensorWord*>(vlhs);
            const auto& rhs = *static_cast<const TensorWord*>(vrhs);
            return ranges::equal(lhs, rhs);
        };
        support->comparison.not_equals
                = +[](const void* vlhs, const void* vrhs) {
                      const auto& lhs = *static_cast<const TensorWord*>(vlhs);
                      const auto& rhs = *static_cast<const TensorWord*>(vrhs);
                      return !ranges::equal(lhs, rhs);
                  };

        support->comparison.less
                = +[](const void* vlhs, const void* vrhs) { return false; };

        support->comparison.less_equal
                = +[](const void* vlhs, const void* vrhs) { return false; };
        support->comparison.greater
                = +[](const void* vlhs, const void* vrhs) { return false; };
        support->comparison.greater_equal
                = +[](const void* vlhs, const void* vrhs) { return false; };

        support->conversions.convert = +[](void* dst, const void* src) {
            auto& dst_word = *static_cast<TensorWord*>(dst);
            const auto& src_word = *static_cast<const TensorWord*>(src);
            dst_word = src_word;
        };
        support->conversions.move_convert = +[](void* dst, void* src) {
            auto& dst_word = *static_cast<TensorWord*>(dst);
            auto& src_word = *static_cast<TensorWord*>(src);
            dst_word = std::move(src_word);
        };
    }

    set_hash_fn(+[](const void* value_ptr) {
        return hash_value(*static_cast<const TensorWord*>(value_ptr));
    });
}

void* TensorWordType::allocate_single() const { return new TensorWord(); }
void TensorWordType::free_single(void* ptr) const
{
    delete static_cast<TensorWord*>(ptr);
}
bool TensorWordType::supports_device(const devices::Device& device
) const noexcept
{
    return device->is_host();
}
void TensorWordType::copy(void* dst, const void* src, dimn_t count) const
{
    const auto* src_begin = static_cast<const TensorWord*>(src);
    const auto* src_end = src_begin + count;
    auto* dst_begin = static_cast<TensorWord*>(dst);

    ranges::copy(src_begin, src_end, dst_begin);
}
void TensorWordType::move(void* dst, void* src, dimn_t count) const
{
    auto* src_begin = static_cast<TensorWord*>(src);
    auto* src_end = src_begin + count;
    auto* dst_begin = static_cast<TensorWord*>(dst);

    ranges::move(src_begin, src_end, dst_begin);
}

void TensorWordType::display(std::ostream& os, const void* ptr) const
{
    const auto& key = *static_cast<const TensorWord*>(ptr);
    auto it = key.begin();
    const auto end = key.end();
    if (it != end) {
        os << *it;
        ++it;
        for (; it != end; ++it) { os << ',' << *it; }
    }
}
devices::TypePtr TensorWordType::get()
{
    static devices::TypePtr type(new TensorWordType);
    return type;
}
