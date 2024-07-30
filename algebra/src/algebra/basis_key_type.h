//
// Created by sam on 08/04/24.
//

#ifndef ROUGHPY_BASIS_KEY_TYPE_H
#define ROUGHPY_BASIS_KEY_TYPE_H

#include <roughpy/devices/core.h>
#include <roughpy/devices/type.h>

#include "roughpy_algebra_export.h"

namespace rpy {
namespace algebra {

class BasisKeyType : public devices::Type
{
public:
    BasisKeyType();

    RPY_NO_DISCARD devices::Buffer
    allocate(devices::Device device, dimn_t count) const override;
    RPY_NO_DISCARD void* allocate_single() const override;
    void free_single(void* ptr) const override;
    RPY_NO_DISCARD bool supports_device(const devices::Device& device
    ) const noexcept override;
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_BASIS_KEY_TYPE_H
