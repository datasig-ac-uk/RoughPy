//
// Created by sam on 8/15/24.
//

#ifndef INDEX_KEY_H
#define INDEX_KEY_H

#include <roughpy/devices/type.h>

namespace rpy {
namespace algebra {

class IndexKeyType : public devices::Type
{

public:
    IndexKeyType();

    RPY_NO_DISCARD void* allocate_single() const override;
    void free_single(void* ptr) const override;
    RPY_NO_DISCARD bool supports_device(const devices::Device& device
    ) const noexcept override;
    void copy(void* dst, const void* src, dimn_t count) const override;
    void move(void* dst, void* src, dimn_t count) const override;
    void display(std::ostream& os, const void* ptr) const override;
    RPY_NO_DISCARD devices::ConstReference zero() const override;
    RPY_NO_DISCARD devices::ConstReference one() const override;

    static devices::TypePtr get();
};

}// namespace algebra
}// namespace rpy

#endif// INDEX_KEY_H
