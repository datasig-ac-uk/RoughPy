// ReSharper disable CppParameterMayBeConstPtrOrRef
#include "tensor_basis.h"

#include <stddef.h>
#include <stdlib.h>

#define RPC_PYCOMPAT_INCLUDE_STRUCTMEMBER 1
#include "py_compat.h"

#include "py_fnv1a_hash.h"

/* clang-format off */
struct _PyTensorBasis {
    PyObject_HEAD//
    int32_t width;
    int32_t depth;
    PyObject* degree_begin;

    Py_hash_t cached_hash;
};
/* clang-format on */

static PyObject* tensor_basis_new(
        PyTypeObject* type,
        PyObject* Py_UNUSED(args),
        PyObject* Py_UNUSED(kwargs)
)
{
    PyTensorBasis* self = (PyTensorBasis*) type->tp_alloc(type, 0);
    if (!self) { return NULL; }

    Py_XSETREF(self->degree_begin, Py_NewRef(Py_None));

    self->cached_hash = -1;

    return (PyObject*) self;
}

static inline PyObject* construct_tensor_db(int32_t width, int32_t depth)
{

    npy_intp const shape[1] = {depth + 2};
    PyObject* arr = PyArray_SimpleNew(1, shape, NPY_INTP);

    if (!arr) { return NULL; }

    npy_intp* data = (npy_intp*) PyArray_DATA((PyArrayObject*) arr);

    data[0] = 0;
    for (npy_intp i = 1; i < depth + 2; ++i) {
        data[i] = 1 + data[i - 1] * width;
    }

    return arr;
}

static int
tensor_basis_init(PyTensorBasis* self, PyObject* args, PyObject* kwargs)
{
    static char* kwords[] = {"width", "depth", "degree_begin", NULL};
    PyObject* degree_begin = NULL;

    if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "ii|O",
                kwords,
                &self->width,
                &self->depth,
                &degree_begin
        )) {
        return -1;
    }

    if (degree_begin) {
        if (!PyArray_Check(degree_begin)) {
            PyErr_SetString(PyExc_TypeError, "expected numpy array");
            return -1;
        }

        PyArrayObject* arr = (PyArrayObject*) degree_begin;

        if (PyArray_NDIM(arr) != 1) {
            PyErr_SetString(PyExc_TypeError, "expected 1d array");
            return -1;
        }

        if (PyArray_TYPE(arr) != NPY_INTP) {
            PyErr_SetString(
                    PyExc_TypeError,
                    "elements of degree begin must be signed pointer-sized "
                    "integers"
            );
            return -1;
        }

        npy_intp size = PyArray_DIM(arr, 0);
        if (size >= 2 && size < self->depth + 2) {
            PyErr_SetString(
                    PyExc_ValueError,
                    "degree begin array must be at least depth+2 elements"
            );
            return -1;
        }

        npy_intp const* data = (npy_intp*) PyArray_DATA(arr);
        if (data[0] != 0 || data[1] != 1) {
            PyErr_SetString(
                    PyExc_ValueError,
                    "first two elements should be 0 and 1"
            );
            return -1;
        }
        if (self->depth >= 1 && data[1] - 1) {
            PyErr_SetString(PyExc_ValueError, "data[2] must match the width");
            return -1;
        }

        Py_XSETREF(self->degree_begin, Py_NewRef(degree_begin));
    } else {
        PyObject* arr = construct_tensor_db(self->width, self->depth);
        if (arr == NULL) { return -1; }
        Py_XSETREF(self->degree_begin, arr);
    }

    return 0;
}

static PyObject* tensor_basis_repr(PyTensorBasis* self)
{
    return PyUnicode_FromFormat(
            "TensorBasis(%i, %i)",
            self->width,
            self->depth
    );
}

static Py_hash_t tensor_basis_hash(PyObject* obj)
{
    PyTensorBasis* self = (PyTensorBasis*) obj;

    // We may need to replace this with an atomic load
    Py_hash_t hash = self->cached_hash;
    if (hash != -1) { return hash; }

    Py_uhash_t state = FNV1A_OFFSET_BASIS;
    state = fnv1a_hash_bytes(state, "TensorBasis", 11);

    state = fnv1a_hash_i32(state, self->width);
    state = fnv1a_hash_i32(state, self->depth);

    // also include the degree-begin data to be sure we get everything right
    const npy_intp db_bytes = ((npy_intp) self->depth + 2) * sizeof(npy_intp);
    const void* db_data = PyArray_DATA((PyArrayObject*) self->degree_begin);
    state = fnv1a_hash_bytes(state, db_data, db_bytes);

    // maybe there are other things to include later.

    hash = fnv1a_finalize_hash(state);

    // leave room for an atomic store later
    self->cached_hash = hash;

    return hash;
}

static PyObject*
tensor_basis_truncate(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char* kwords[] = {"new_depth", NULL};
    int32_t new_depth;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwords, &new_depth)) {
        return NULL;
    }

    if (new_depth < 0) {
        PyErr_SetString(PyExc_ValueError, "new_depth must be non-negative");
        return NULL;
    }

    PyTensorBasis* self_ = (PyTensorBasis*) self;

    if (new_depth >= self_->depth) {
        PyErr_SetString(PyExc_ValueError, "new_depth must be less than depth");
        return NULL;
    }

    PyTensorBasis* new_obj
            = PyObject_NewVar(PyTensorBasis, &PyTensorBasis_Type, 0);
    if (new_obj == NULL) { return NULL; }

    new_obj->width = self_->width;
    new_obj->depth = new_depth;

    // We want to do this, but the Py_NewRef was added in 3.10
    // new_obj->degree_begin = Py_NewRef(self_->degree_begin);
    Py_INCREF(self_->degree_begin);
    new_obj->degree_begin = self_->degree_begin;

    return (PyObject*) new_obj;
}

PyObject* tensor_basis_size(PyObject* self, PyObject* Py_UNUSED(arg))
{
    PyTensorBasis* self_ = (PyTensorBasis*) self;
    npy_intp* data = PyArray_DATA((PyArrayObject*) self_->degree_begin);
    return PyLong_FromLong(data[self_->depth + 1]);
}

static void tensor_basis_dealloc(PyTensorBasis* self)
{
    Py_XDECREF(self->degree_begin);
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject*
tensor_basis_test_equal(PyTensorBasis* left, PyTensorBasis* right, long success)
{
    if (left->width != right->width) { return PyBool_FromLong(!success); }
    if (left->depth != right->depth) { return PyBool_FromLong(!success); }

    const npy_intp* l_db_data
            = PyArray_DATA((PyArrayObject*) left->degree_begin);
    const npy_intp* r_db_data
            = PyArray_DATA((PyArrayObject*) right->degree_begin);

    for (int32_t d = 0; d < left->depth + 2; ++d) {
        if (l_db_data[d] != r_db_data[d]) { return PyBool_FromLong(!success); }
    }

    return PyBool_FromLong(success);
}

static PyObject*
tensor_basis_richcompare(PyObject* self, PyObject* other, int op)
{
    if (!PyTensorBasis_Check(self) || !PyTensorBasis_Check(other)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    PyTensorBasis* left = (PyTensorBasis*) self;
    PyTensorBasis* right = (PyTensorBasis*) other;

    switch (op) {
        case Py_EQ: return tensor_basis_test_equal(left, right, 1);
        case Py_NE: return tensor_basis_test_equal(left, right, 0);
        case Py_LT:
        case Py_LE:
        case Py_GT:
        case Py_GE:
        default: break;
    }

    Py_RETURN_NOTIMPLEMENTED;
}

static PyMemberDef PyTensorBasis_members[] = {
        {"width",
         Py_T_INT, offsetof(PyTensorBasis, width),
         Py_READONLY, "width of the basis"},
        {"depth",
         Py_T_INT, offsetof(PyTensorBasis, depth),
         Py_READONLY, "depth of the basis"},
        {"degree_begin",
         Py_T_OBJECT_EX, offsetof(PyTensorBasis, degree_begin),
         0, "degree_begin"},
        {NULL}
};

static PyMethodDef PyTensorBasis_methods[] = {
        {"truncate",
         (PyCFunction) tensor_basis_truncate,
         METH_VARARGS | METH_KEYWORDS,
         "Truncate the basis to a smaller depth."},
        {"size",
         (PyCFunction) tensor_basis_size,
         METH_NOARGS, "Size of the basis."},
        {NULL}
};

/* clang-format off */
PyTypeObject PyTensorBasis_Type = {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = RPY_CPT_TYPE_NAME(TensorBasis),
        .tp_basicsize = sizeof(PyTensorBasis),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "TensorBasis",
        .tp_methods = PyTensorBasis_methods,
        .tp_members = PyTensorBasis_members,
        .tp_new = tensor_basis_new,
        .tp_init = (initproc) tensor_basis_init,
        .tp_dealloc = (destructor) tensor_basis_dealloc,
        .tp_repr = (reprfunc) tensor_basis_repr,
        .tp_hash = (hashfunc) tensor_basis_hash,
        .tp_richcompare = (richcmpfunc) tensor_basis_richcompare,
};
/* clang-format on */

int init_tensor_basis(PyObject* module)
{
    if (PyType_Ready(&PyTensorBasis_Type) < 0) { return -1; }

    if (PyModule_AddObjectRef(
                module,
                "TensorBasis",
                (PyObject*) &PyTensorBasis_Type
        )
        < 0) {
        return -1;
    }

    return 0;
}

PyTensorBasis* PyTensorBasis_get(int32_t width, int32_t depth)
{
    if (width <= 0 || depth <= 0) {
        PyErr_SetString(PyExc_ValueError, "width must be positive");
        return NULL;
    }

    PyTensorBasis* self
            = (PyTensorBasis*)
                      PyTensorBasis_Type.tp_alloc(&PyTensorBasis_Type, 0);
    if (!self) { return NULL; }

    self->width = width;
    self->depth = depth;

    PyObject* db = construct_tensor_db(width, depth);
    if (db == NULL) {
        Py_DECREF(self);
        return NULL;
    }

    Py_XSETREF(self->degree_begin, db);
    return self;
}

/******************************************************************************
 * External methods
 ******************************************************************************/
int32_t PyTensorBasis_width(PyTensorBasis* basis) { return basis->width; }
int32_t PyTensorBasis_depth(PyTensorBasis* basis) { return basis->depth; }
PyArrayObject* PyTensorBasis_degree_begin(PyTensorBasis* basis)
{
    return (PyArrayObject*) basis->degree_begin;
}
