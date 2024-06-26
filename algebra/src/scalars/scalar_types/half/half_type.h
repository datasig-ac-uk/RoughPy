//
// Created by sam on 24/06/24.
//

#ifndef ROUGHPY_SCALAR_IMPLEMENTATIONS_HALF_TYPE_H
#define ROUGHPY_SCALAR_IMPLEMENTATIONS_HALF_TYPE_H

#include <roughpy/devices/type.h>


namespace rpy {
namespace scalars {
namespace implementations {

class HalfType : public devices::Type {

    HalfType();
public:
    RPY_NO_DISCARD devices::Buffer
    allocate(devices::Device device, dimn_t count) const override;
    RPY_NO_DISCARD void* allocate_single() const override;
    void free_single(void* ptr) const override;
    RPY_NO_DISCARD bool supports_device(const devices::Device& device
    ) const noexcept override;
    RPY_NO_DISCARD bool convertible_to(const Type& dest_type
    ) const noexcept override;
    RPY_NO_DISCARD bool convertible_from(const Type& src_type
    ) const noexcept override;
    RPY_NO_DISCARD devices::TypeComparison compare_with(const Type& other
    ) const noexcept override;
    void copy(void* dst, const void* src, dimn_t count) const override;
    void move(void* dst, void* src, dimn_t count) const override;
    void display(std::ostream& os, const void* ptr) const override;
    RPY_NO_DISCARD devices::ConstReference zero() const override;
    RPY_NO_DISCARD devices::ConstReference one() const override;
    RPY_NO_DISCARD devices::ConstReference mone() const override;
};

} // implementations
} // scalars
} // rpy

#endif //ROUGHPY_SCALAR_IMPLEMENTATIONS_HALF_TYPE_H
