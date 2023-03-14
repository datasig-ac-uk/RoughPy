#include "lie_key_iterator.h"

#include <roughpy/algebra/basis.h>
#include <roughpy/algebra/context.h>



using namespace rpy;
using namespace pybind11::literals;


static const char *LKEY_ITERATOR_DOC = R"eadoc(Iterator over range of Hall set members.
)eadoc";

python::PyLieKeyIterator::PyLieKeyIterator(const PyContext& ctx, key_type current, key_type end)
    : p_ctx(ctx.get_context()), m_current(current), m_end(end)
{

}

static python::PyLieKey to_py_lie_key(key_type k, const algebra::Basis &lbasis) {
    auto width = lbasis.width().value();

    if (lbasis.letter(k)) {
        return python::PyLieKey(width, k);
    }

    auto lparent = lbasis.lparent(k).value();
    auto rparent = lbasis.rparent(k).value();

    if (lbasis.letter(lparent) && lbasis.letter(rparent)) {
        return python::PyLieKey(lbasis.width().value(), lparent, rparent);
    }
    if (lbasis.letter(lparent)) {
        return python::PyLieKey(width, lparent, to_py_lie_key(rparent, lbasis));
    }
    return python::PyLieKey(width,
                      to_py_lie_key(lparent, lbasis),
                      to_py_lie_key(rparent, lbasis));
}

python::PyLieKey python::PyLieKeyIterator::next() {
    if (m_current > m_end) {
        throw py::stop_iteration();
    }
    auto current = m_current;
    ++m_current;
    return to_py_lie_key(current, p_ctx->get_lie_basis());
}

void python::init_lie_key_iterator(py::module_ &m) {
    py::class_<PyLieKeyIterator> klass(m, "LieKeyIterator", LKEY_ITERATOR_DOC);
    klass.def(py::init<PyContext>(), "context"_a);
    klass.def(py::init<PyContext, key_type>(), "context"_a, "start_key"_a);
    klass.def(py::init<PyContext, key_type, key_type>(), "context"_a, "start_key"_a, "end_key"_a);

    klass.def("__iter__", [](PyLieKeyIterator &self) { return self; });
    klass.def("__next__", &PyLieKeyIterator::next);
}
