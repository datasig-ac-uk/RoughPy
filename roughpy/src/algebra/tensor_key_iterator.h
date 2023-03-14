#ifndef RPY_PY_ALGEBRA_TENSOR_KEY_ITERATOR_H_
#define RPY_PY_ALGEBRA_TENSOR_KEY_ITERATOR_H_

#include "roughpy_module.h"

#include <limits>

#include "context.h"
#include "tensor_key.h"


namespace rpy {
namespace python {

class PyTensorKeyIterator {
    key_type m_current;
    key_type m_end;
    deg_t m_width;
    deg_t m_depth;

public:

    PyTensorKeyIterator(deg_t width,
                        deg_t depth,
                        key_type current=0,
                        key_type end=std::numeric_limits<key_type>::max());

    PyTensorKey next();

};


void init_tensor_key_iterator(py::module_& m);


} // namespace python
} // namespace rpy

#endif // RPY_PY_ALGEBRA_TENSOR_KEY_ITERATOR_H_
