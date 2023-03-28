//
// Created by user on 28/03/23.
//

#include "algebra_iterator.h"

#include <roughpy/core/macros.h>
#include <roughpy/algebra/algebra_iterator.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/algebra/shuffle_tensor.h>

#include "lie_key.h"
#include "tensor_key.h"

using namespace rpy;
using namespace rpy::algebra;


template <typename Algebra, typename KeyType>
static void init_iterator(py::module_& m, const char* iter_name) {

    py::class_<AlgebraIteratorItem<Algebra>> klass(m, iter_name);

    klass.def("key", [](const AlgebraIteratorItem<Algebra>& item) {
        return KeyType(item.basis(), item.key());
    });
    klass.def("scalar", &AlgebraIteratorItem<Algebra>::value);

}

#define MAKE_ITERATOR_TYPE(ALG, KEY) init_iterator<ALG, KEY>(m, RPY_STRINGIFY(ALG ## IteratorItem))

void python::init_algebra_iterator(py::module_ &m) {

    MAKE_ITERATOR_TYPE(Lie, PyLieKey);
    MAKE_ITERATOR_TYPE(FreeTensor, PyTensorKey);
    MAKE_ITERATOR_TYPE(ShuffleTensor, PyTensorKey);

}
#undef MAKE_ITERATOR_TYPE
