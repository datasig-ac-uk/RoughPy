#include "basis.h"

#include <roughpy/algebra/basis.h>
#include <roughpy/algebra/tensor_basis.h>
#include <roughpy/algebra/lie_basis.h>

#include "lie_key.h"
#include "tensor_key.h"

using namespace rpy;
using namespace rpy::algebra;
using namespace pybind11::literals;

template <typename T, typename K>
static void wordlike_basis_setup(py::module_& m, const char* name) {

    py::class_<T> basis(m, name);

    basis.def_property_readonly("width", &T::width);
    basis.def_property_readonly("depth", &T::depth);
    basis.def_property_readonly("dimension", &T::dimension);

    basis.def("index_to_key", [](const T& self, dimn_t index) {
             return K(self, self.index_to_key(index));
         }, "index"_a);
    basis.def("key_to_index", [](const T& self, const python::PyLieKey& key) {
        return self.key_to_index(0);
    }, "key"_a );

    basis.def("parents", [](const T& self, const K& key) {
        return self.parents(0);
    }, "key"_a);
    basis.def("size", &T::size);



}


void python::init_basis(py::module_ &m) {

    wordlike_basis_setup<TensorBasis, PyTensorKey>(m, "TensorBasis");
    wordlike_basis_setup<LieBasis, PyLieKey>(m, "LieBasis");

}
