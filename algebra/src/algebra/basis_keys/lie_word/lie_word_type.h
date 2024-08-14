//
// Created by sam on 8/13/24.
//

#ifndef LIE_WORD_TYPE_H
#define LIE_WORD_TYPE_H

#include <roughpy/devices/type.h>


namespace rpy {
namespace algebra {

class LieWordType : public devices::Type {

public:
    LieWordType();

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
};

} // algebra
} // rpy

#endif //LIE_WORD_TYPE_H
