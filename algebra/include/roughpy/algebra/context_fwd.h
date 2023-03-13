#ifndef ROUGHPY_ALGEBRA_CONTEXT_FWD_H_
#define ROUGHPY_ALGEBRA_CONTEXT_FWD_H_

#include "algebra_fwd.h"
#include <stdexcept>

#include <boost/smart_ptr/intrusive_ptr.hpp>

#define RPY_MAKE_VTYPE_SWITCH(VTYPE)                            \
    switch (VTYPE) {                                            \
        case VectorType::Dense:                                 \
            return RPY_SWITCH_FN(VectorType::Dense);            \
        case VectorType::Sparse:                                \
            return RPY_SWITCH_FN(VectorType::Sparse);           \
        default:                                                \
            throw std::invalid_argument("invalid vector type"); \
    }

#define RPY_MAKE_ALGTYPE_SWITCH(ALGTYPE)                         \
    switch (ALGTYPE) {                                           \
        case AlgebraType::FreeTensor:                            \
            return RPY_SWITCH_FN(AlgebraType::FreeTensor);       \
        case AlgebraType::Lie:                                   \
            return RPY_SWITCH_FN(AlgebraType::Lie);              \
        case AlgebraType::ShuffleTensor:                         \
            return RPY_SWITCH_FN(AlgebraType::ShuffleTensor);    \
        default:                                                 \
            throw std::invalid_argument("invalid algebra type"); \
    }

namespace rpy {
namespace algebra {

struct SignatureData;
struct DerivativeComputeInfo;
class VectorConstructionData;
class ContextBase;
class Context;

using base_context_pointer = boost::intrusive_ptr<const ContextBase>;
using context_pointer = boost::intrusive_ptr<const Context>;

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_CONTEXT_FWD_H_
