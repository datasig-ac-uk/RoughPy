#ifndef ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_BUNDLE_INTERFACE_H_
#define ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_BUNDLE_INTERFACE_H_

#include <roughpy/core/macros.h>

#include "shuffle_tensor_interface.h"
#include "algebra_bundle_interface.h"

namespace rpy { namespace algebra {

RPY_TEMPLATE_EXTERN template class RPY_EXPORT_TEMPLATE
BundleInterface<ShuffleTensorBundle, ShuffleTensor, ShuffleTensor>;

class ROUGHPY_ALGEBRA_EXPORT ShuffleTensorBundleInterface
    : public BundleInterface<ShuffleTensorBundle, ShuffleTensor, ShuffleTensor>
{
};






}}

#endif // ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_BUNDLE_INTERFACE_H_
