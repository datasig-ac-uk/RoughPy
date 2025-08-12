#include "tensor_basis.h"

#include <stdlib.h>
#include <structmember.h>

static PyObject* tensor_basis_new(PyTypeObject* type,
                                  PyObject* Py_UNUSED(args),
                                  PyObject* Py_UNUSED(kwargs))
{
    PyTensorBasis* self = (PyTensorBasis*) type->tp_alloc(type, 0);
    if (!self) { return NULL; }

    Py_XSETREF(self->degree_begin, Py_NewRef(Py_None));

    return (PyObject*) self;
}


static int tensor_basis_init(PyTensorBasis* self,
                             PyObject* args,
                             PyObject* kwargs)
{
    static char* kwords[] = {"width", "depth", "degree_begin", NULL};
    PyObject* degree_begin = NULL;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "ii|O",
                                     kwords,
                                     &self->width,
                                     &self->depth,
                                     &degree_begin)) { return -1; }

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
            PyErr_SetString(PyExc_TypeError,
                            "elements of degree begin must be signed pointer-sized integers");
            return -1;
        }

        npy_intp size = PyArray_DIM(arr, 0);
        if (size >= 2 && size < self->depth + 2) {
            PyErr_SetString(PyExc_ValueError,
                            "degree begin array must be at least depth+2 elements");
            return -1;
        }

        npy_intp const* data = (npy_intp*) PyArray_DATA(arr);
        if (data[0] != 0 || data[1] != 1) {
            PyErr_SetString(PyExc_ValueError,
                            "first two elements should be 0 and 1");
            return -1;
        }
        if (self->depth >= 1 && data[2] - 1) {
            PyErr_SetString(PyExc_ValueError, "data[2] must match the width");
            return -1;
        }

        Py_XSETREF(self->degree_begin, Py_NewRef(degree_begin));
    } else {
        npy_intp const shape[1] = {self->depth + 2};
        PyObject* arr = PyArray_SimpleNew(1, shape, NPY_INTP);
        // Py_XSETREF(self->degree_begin, PyArray_SimpleNew(1, shape, NPY_INTP));

        if (!arr) { return -1; }

        npy_intp* data = (npy_intp*) PyArray_DATA(
            (PyArrayObject*) arr);

        data[0] = 0;
        for (npy_intp i = 1; i < self->depth + 2; ++i) {
            data[i] = 1 + data[i - 1] * self->width;
        }

        Py_XSETREF(self->degree_begin, arr);
    }

    return 0;
}

static PyObject* tensor_basis_repr(PyTensorBasis* self)
{
    return PyUnicode_FromFormat("TensorBasis(%i, %i)",
                                self->width,
                                self->depth);
}


static PyObject* tensor_basis_truncate(PyObject* self,
                                       PyObject* args,
                                       PyObject* kwargs)
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

    PyTensorBasis* new_obj = PyObject_NewVar(PyTensorBasis,
                                             &PyTensorBasis_Type,
                                             0);
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

static PyMemberDef PyTensorBasis_members[] = {
        {"width", Py_T_INT, offsetof(PyTensorBasis, width),
         READONLY, "width of the basis"},
        {"depth", Py_T_INT, offsetof(PyTensorBasis, depth),
         READONLY, "depth of the basis"},
        {"degree_begin", Py_T_OBJECT_EX,offsetof(PyTensorBasis, degree_begin),
         0, "degree_begin"},
        {NULL}
};

static PyMethodDef PyTensorBasis_methods[] = {
        {"truncate", (PyCFunction) tensor_basis_truncate,
         METH_VARARGS | METH_KEYWORDS,
         "Truncate the basis to a smaller depth."},
        {"size", (PyCFunction) tensor_basis_size, METH_NOARGS,
         "Size of the basis."},
        {NULL}
};


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
        .tp_repr = (reprfunc) tensor_basis_repr
};


int init_tensor_basis(PyObject* module)
{
    if (PyType_Ready(&PyTensorBasis_Type) < 0) { return -1; }

    if (PyModule_AddObjectRef(module,
                              "TensorBasis",
                              (PyObject*) &PyTensorBasis_Type) < 0) {
        return -1;
    }

    return 0;
}