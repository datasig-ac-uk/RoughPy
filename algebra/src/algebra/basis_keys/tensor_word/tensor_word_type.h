//
// Created by sam on 8/12/24.
//

#ifndef TENSOR_WORD_TYPE_H
#define TENSOR_WORD_TYPE_H

#include <roughpy/devices/type.h>

namespace rpy {
namespace algebra {

class TensorWordType final : public devices::Type
{

public:
    TensorWordType();

    RPY_NO_DISCARD void* allocate_single() const override;
    void free_single(void* ptr) const override;
    RPY_NO_DISCARD bool supports_device(const devices::Device& device
    ) const noexcept override;
    void copy(void* dst, const void* src, dimn_t count) const override;
    void move(void* dst, void* src, dimn_t count) const override;
    void display(std::ostream& os, const void* ptr) const override;
};

}// namespace algebra
}// namespace rpy

#endif// TENSOR_WORD_TYPE_H
