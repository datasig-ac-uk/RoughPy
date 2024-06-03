//
// Created by sam on 3/30/24.
//

#ifndef AP_RATIONAL_TYPE_H
#define AP_RATIONAL_TYPE_H

#include <roughpy/device_support/fundamental_type.h>
#include "scalar_implementations/arbitrary_precision_rational.h"

namespace rpy {
namespace devices {

class RPY_LOCAL APRationalType : public Type {

public:

    APRationalType();

    RPY_NO_DISCARD Buffer allocate(Device device, dimn_t count) const override;
    RPY_NO_DISCARD void* allocate_single() const override;
    void free_single(void* ptr) const override;
    bool supports_device(const Device& device) const noexcept override;

    RPY_NO_DISCARD static const APRationalType* get() noexcept;

    void display(std::ostream& os, const void* ptr) const override;

    void copy(void* dst, const void* src, dimn_t count) const override;
    void move(void* dst, void* src, dimn_t count) const override;
    RPY_NO_DISCARD ConstReference zero() const override;
    RPY_NO_DISCARD ConstReference one() const override;
    RPY_NO_DISCARD ConstReference mone() const override;
};


}
}

#endif //AP_RATIONAL_TYPE_H
