#ifndef ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_H_

#include "algebra_fwd.h"

namespace rpy {
namespace algebra {


template <typename Base, typename Fibre>
class ROUGHPY_ALGEBRA_EXPORT BundleInterface : public Base::interface_t {
public:

    using base_t = Base;
    using fibre_t = Fibre;
    using base_interface_t = typename Base::interface_t;
    using fibre_interface_t = typename Fibre::interface_t;

    virtual Fibre fibre() = 0;
};


}
}
#endif // ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_H_
