#include "basis.h"

#include <roughpy/algebra/basis.h>
#include <roughpy/algebra/tensor_basis.h>
#include <roughpy/algebra/lie_basis.h>

using namespace rpy;
using namespace rpy::algebra;

void python::init_basis(py::module_ &m) {

    py::class_<TensorBasis> tensor_basis(m, "TensorBasis");
    py::class_<LieBasis> lie_basis(m, "LieBasis");
    // TODO: add method definitions

}
