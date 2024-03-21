//
// Created by sam on 3/19/24.
//

#ifndef ROUGHPY_AUGMENTED_BASIS_H
#define ROUGHPY_AUGMENTED_BASIS_H

#include "basis.h"

#include <functional>
namespace rpy {
namespace algebra {

struct OrderedBasisProperties {
};

struct GradedBasisProperties {
};

struct WordlikeBasisProperties {
};

class AugmentedBasis : public Basis
{
    BasisPointer p_base;
    std::unique_ptr<const OrderedBasisProperties> p_ordering;
    std::unique_ptr<const GradedBasisProperties> p_grading;
    std::unique_ptr<const WordlikeBasisProperties> p_wordlike;

public:
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_AUGMENTED_BASIS_H
