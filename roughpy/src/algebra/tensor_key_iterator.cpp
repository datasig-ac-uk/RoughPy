#include "tensor_key_iterator.h"

#include <roughpy/algebra/basis.h>
#include <roughpy/algebra/context.h>


using namespace rpy;
using namespace pybind11::literals;

static const char *TKEY_ITERATOR_DOC = R"eadoc(Iterator over tensor words.
)eadoc";

python::PyTensorKeyIterator::PyTensorKeyIterator(deg_t width, deg_t depth, key_type current, key_type end)
    : m_width(width), m_depth(depth), m_current(current), m_end(end)
{
}
python::PyTensorKey python::PyTensorKeyIterator::next() {
    if (m_current >= m_end) {
        throw py::stop_iteration();
    }
    auto current = m_current;
    ++m_current;
    return PyTensorKey(current, m_width, m_depth);
}

void python::init_tensor_key_iterator(py::module_ &m) {

    py::class_<PyTensorKeyIterator> klass(m, "TensorKeyIterator", TKEY_ITERATOR_DOC);

    klass.def(py::init([](const PyTensorKey &start_key) {
                return PyTensorKeyIterator(start_key.width(), start_key.depth(), static_cast<key_type>(start_key));
              }),
              "start_key"_a);
    klass.def(py::init([](const PyTensorKey &start_key, const PyTensorKey &end_key) {
                return PyTensorKeyIterator(start_key.width(), start_key.depth(), static_cast<key_type>(start_key), static_cast<key_type>(end_key));
              }),
              "start_key"_a, "end_key"_a);
    klass.def("__iter__", [](PyTensorKeyIterator &self) { return self; });
    klass.def("__next__", &PyTensorKeyIterator::next);
}
