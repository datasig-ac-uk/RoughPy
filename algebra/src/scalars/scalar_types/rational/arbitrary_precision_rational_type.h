//
// Created by sam on 24/06/24.
//

#ifndef ARBITRARY_PRECISION_RATIONAL_TYPE_H
#define ARBITRARY_PRECISION_RATIONAL_TYPE_H


#include <roughpy/core/types.h>
#include <roughpy/devices/type.h>
#include <roughpy/devices/buffer.h>

#include "arbitrary_precision_rational.h"

namespace rpy {
namespace scalars {
namespace implementations {

class RPY_LOCAL ArbitraryPrecisionRationalType : public devices::Type{

    ArbitraryPrecisionRationalType();
public:

    RPY_NO_DISCARD devices::Buffer
    allocate(devices::Device device, dimn_t count) const override;
    RPY_NO_DISCARD void* allocate_single() const override;
    void free_single(void* ptr) const override;
    RPY_NO_DISCARD bool supports_device(const devices::Device& device
    ) const noexcept override;
    RPY_NO_DISCARD devices::TypeComparison compare_with(const Type& other
    ) const noexcept override;
    void copy(void* dst, const void* src, dimn_t count) const override;
    void move(void* dst, void* src, dimn_t count) const override;
    void display(std::ostream& os, const void* ptr) const override;
    RPY_NO_DISCARD devices::ConstReference zero() const override;
    RPY_NO_DISCARD devices::ConstReference one() const override;
    RPY_NO_DISCARD devices::ConstReference mone() const override;


    static const ArbitraryPrecisionRationalType* get() ;
};

} // implementations
} // scalars
} // rpy

#endif //ARBITRARY_PRECISION_RATIONAL_TYPE_H
