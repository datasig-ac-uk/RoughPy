//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_INTO_FROM_H
#define ROUGHPY_GENERICS_INTERNAL_INTO_FROM_H


#include <memory>
#include <utility>

namespace rpy {
namespace generics {

class IntoTrait {};
class FromTrait {};

class IntoFrom final : public IntoTrait {
    std::unique_ptr<const FromTrait> p_from;
 public:

    explicit IntoFrom(std::unique_ptr<const FromTrait> from)
        : p_from(std::move(from))
    {}



};

} // generics
} // rpy

#endif //ROUGHPY_GENERICS_INTERNAL_INTO_FROM_H
