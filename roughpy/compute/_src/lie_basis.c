#include "lie_basis.h"

#include <stddef.h>
#include <structmember.h>


static PyObject* lie_basis_new(
    PyTypeObject* type,
    PyObject* Py_UNUSED(args),
    PyObject* Py_UNUSED(kwargs))
{
    PyLieBasis* self = (PyLieBasis*) type->tp_alloc(type, 0);
    if (!self) { return NULL; }

    Py_XSETREF(self->degree_begin, Py_NewRef(Py_None));
    Py_XSETREF(self->data, Py_NewRef(Py_None));

    return (PyObject*) self;
}

static void lie_basis_dealloc(PyLieBasis* self)
{
    Py_XDECREF(self->degree_begin);
    Py_XDECREF(self->data);
    Py_TYPE(self)->tp_free((PyObject*) self);
}


static int construct_lie_basis(PyLieBasis* self)
{
    PyObject* data = NULL;
    PyObject* degree_begin = NULL;
    int ret = -1;

    /*
     * For start, let's set the size of the lie basis to
     * the size of the tensor algebra, which is definitely
     * large enough to hold the data. We can resize down
     * later.
     */
    npy_intp alloc_size = 1;
    for (npy_intp i=1; i<=self->depth; ++i) {
        alloc_size = 1 + alloc_size * self->width;
    }

    npy_intp degree_begin_shape[1] = { self->depth + 2};
    degree_begin = PyArray_SimpleNew(1, degree_begin_shape, NPY_INTP);

    if (degree_begin == NULL) {
        goto cleanup;
    }

    npy_intp data_shape[2] = { alloc_size, 2 };
    data = PyArray_SimpleNew(2, data_shape, NPY_INTP);

    if (data == NULL) {
        goto cleanup;
    }

    npy_intp* data_ptr = (npy_intp*) PyArray_DATA((PyArrayObject*) data);
    npy_intp* db_ptr = (npy_intp*) PyArray_DATA((PyArrayObject*) degree_begin);


    /*
     * Now we build the actual hall set. This is purely
     * computational and doesn't require interaction with
     * any Python objects so we can release the GIL.
     */
    Py_BEGIN_ALLOW_THREADS;
    // Only the "god element" has degree 0
    db_ptr[0] = 0;

    // The "god element" is the first
    data_ptr[0] = 0;
    data_ptr[1] = 0;

    npy_intp size = 1;

    // assign the letters first
    if (self->depth > 0) {

        // letters start at index 1
        db_ptr[1] = 1;

        for (npy_intp letter=1; letter<=self->width; ++letter) {
            data_ptr[2*letter] = 0; // data[letter, 0]
            data_ptr[2*letter+1] = letter; // data[letter, 1]
        }

        size += self->width;
        db_ptr[2] = size;
    }

    for (npy_intp degree=2; degree<=self->depth; ++degree) {
        for (npy_intp left_degree=1; 2*left_degree <= degree; ++left_degree) {
            npy_intp right_degree = degree - left_degree;
            npy_intp i_lower = db_ptr[left_degree];
            npy_intp i_upper = db_ptr[left_degree + 1];
            npy_intp j_lower = db_ptr[right_degree];
            npy_intp j_upper = db_ptr[right_degree + 1];

            for (npy_intp i=i_lower; i<i_upper; ++i) {
                npy_intp j_start = (i+1 > j_lower) ? i+1 : j_lower;
                for (npy_intp j=j_start; j<j_upper; ++j) {
                    if (data_ptr[2*j] <= i) {
                        data_ptr[2*size] = i;
                        data_ptr[2*size+1] = j;
                        ++size;
                    }
                }
            }

            db_ptr[degree+1] = size;
        }
    }

    Py_END_ALLOW_THREADS;

    // data_shape[0] = size;
    PyArray_Dims dims = {
        data_shape, 2
    };

    PyObject* tmp = PyArray_Newshape((PyArrayObject*) data, &dims, NPY_CORDER);
    if (tmp == NULL) {
        Py_XDECREF(tmp);
        goto cleanup;
    }
    Py_XDECREF(self->data);
    self->data = tmp;
    tmp = NULL;


    // Move the degree_begin data into the struct;
    Py_XDECREF(self->degree_begin);
    self->degree_begin = degree_begin;
    degree_begin = NULL;

    ret = 0;

    cleanup:
    Py_XDECREF(degree_begin);
    Py_XDECREF(data);

    return ret;
}

static int check_data_and_db(PyLieBasis* self, PyObject* data, PyObject* degree_begin)
{
    if (!PyArray_Check(data)) {
        PyErr_SetString(PyExc_TypeError, "expected numpy array for data argument");
        return -1;
    }

    if (!PyArray_Check(degree_begin)) {
        PyErr_SetString(PyExc_TypeError, "expected numpy array for degree_begin argument");
        return -1;
    }

    PyArrayObject* data_arr = (PyArrayObject*) data;
    PyArrayObject* db_arr = (PyArrayObject*) degree_begin;

    if (PyArray_TYPE(data_arr) != NPY_INTP) {
        PyErr_SetString(PyExc_ValueError, "data must be (pointer-sized) integers");
        return -1;
    }

    if (PyArray_TYPE(db_arr) != NPY_INTP) {
        PyErr_SetString(PyExc_ValueError, "degree_begin must be (pointer-sized) integers");
        return -1;
    }

    if (PyArray_NDIM(data_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "expected 2-dimensional array for data");
        return -1;
    }

    if (PyArray_DIM(data_arr, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "expected data to of shape (N, 2)");
        return -1;
    }

    if (PyArray_NDIM(db_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "expected 1-dimensional array for db");
        return -1;
    }

    if (PyArray_DIM(db_arr, 0) < self->depth + 2) {
        PyErr_SetString(PyExc_ValueError, "degree_begin array must be contain at least depth + 2 elements");
        return -1;
    }

    npy_intp size = *(npy_intp*) PyArray_GETPTR1(db_arr, self->depth + 1);

    if (PyArray_DIM(data_arr, 0) < size) {
        PyErr_SetString(PyExc_ValueError, "mismatch in size between data and degree_begin arrays");
        return -1;
    }

    return 0;
}


static int lie_basis_init(PyLieBasis* self, PyObject* args, PyObject* kwargs)
{
    static char* kwords[] = {"width", "depth", "degree_begin", "data", NULL};
    PyObject* degree_begin = NULL;
    PyObject* data = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|OO", kwords, &self->width, &self->depth, &degree_begin, &data)) {
        return -1;
    }

    if (data == NULL || degree_begin == NULL)  {
        if (construct_lie_basis(self) < 0) {
            return -1;
        }
    } else {
        // Do some basic sanity checks to make sure
        if (check_data_and_db(self, data, degree_begin) < 0) {
            return -1;
        }

        PyObject* tmp = self->data;
        Py_INCREF(data);
        self->data = data;
        Py_XDECREF(tmp);

        tmp = self->degree_begin;
        Py_INCREF(degree_begin);
        self->degree_begin = degree_begin;
        Py_XDECREF(tmp);
    }

    return 0;
}

static PyObject* lie_basis_repr(PyLieBasis* self)
{
    return PyUnicode_FromFormat("LieBasis(%i, %i)",
                                self->width,
                                self->depth);
}


static PyMemberDef PyLieBasis_members[] = {

        {"width", Py_T_INT, offsetof(PyLieBasis, width), READONLY,
         "width of the basis"},
        {"depth", Py_T_INT, offsetof(PyLieBasis, depth), READONLY,
         "depth of the basis"},
        {"degree_begin", Py_T_OBJECT_EX,offsetof(PyLieBasis, degree_begin),
         READONLY, "array of offsets for each degree"},
        {"data", Py_T_OBJECT_EX, offsetof(PyLieBasis, data), 0, "basis data"},
        {NULL}
};


static PyObject* lie_basis_size(PyObject* self, PyObject* Py_UNUSED(arg))
{
    PyLieBasis* self_ = (PyLieBasis*) self;
    if (Py_IsNone(self_->degree_begin)) {
        PyErr_SetString(PyExc_RuntimeError, "degree_begin is None");
        return NULL;
    }
    npy_intp size = *(npy_intp*) PyArray_GETPTR1(
        (PyArrayObject*) self_->degree_begin,
        self_->depth+1);
    return PyLong_FromLong(size);
}

PyMethodDef PyLieBasis_methods[] = {
        {"size", lie_basis_size, METH_NOARGS, "get the size of the Lie basis"},
        {NULL}
};

PyTypeObject PyLieBasis_Type = {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "roughpy.compute.LieBasis",
        .tp_basicsize = sizeof(PyLieBasis),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT,
        .tp_doc = "LieBasis",
        .tp_methods = PyLieBasis_methods,
        .tp_members = PyLieBasis_members,
        .tp_init = (initproc) lie_basis_init,
        .tp_dealloc = (destructor) lie_basis_dealloc,
        .tp_repr = (reprfunc) lie_basis_repr,
        .tp_new = (newfunc) lie_basis_new,
};

int init_lie_basis(PyObject* module)
{
    if (PyType_Ready(&PyLieBasis_Type) < 0) { return -1; }

    PyModule_AddObjectRef(module, "LieBasis", (PyObject*) &PyLieBasis_Type);

    return 0;
}