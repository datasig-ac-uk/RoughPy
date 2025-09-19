#ifndef ROUGHPY_COMPUTE__SRC_LIE_BASIS_H
#define ROUGHPY_COMPUTE__SRC_LIE_BASIS_H


#include "py_headers.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct _LieWord {
    npy_intp letters[2];
} LieWord;

typedef struct _PyLieBasis PyLieBasis;


extern PyTypeObject PyLieBasis_Type;

/*
 * At the moment, the Hall set we construct obeys the ordering where both
 * left and right keys are kept in lexicographical order. In the future, we
 * might want to allow other bases where this invariant is not satisfied. In
 * that case, we either need a different total ordering on words such that
 * basis->data is in order with respect to this ordering OR fall back to a
 * linear search instead.
 */

// lexicographic order on words
static inline int hall_word_less(const npy_intp* lhs, const npy_intp* rhs)
{
    return lhs[0] < rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1]);
}

static inline int hall_word_equal(const npy_intp* lhs, const npy_intp* rhs)
{
    return lhs[0] == rhs[0] && lhs[1] == rhs[1];
}

PyObject *get_l2t_matrix(PyObject *basis, PyObject *dtype_obj);

PyObject *get_t2l_matrix(PyObject *basis, PyObject *dtype_obj);

static inline int PyLieBasis_Check(PyObject *obj) {
    return PyObject_TypeCheck(obj, &PyLieBasis_Type);
}

RPY_NO_EXPORT
int32_t PyLieBasis_width(PyLieBasis *basis);

RPY_NO_EXPORT
int32_t PyLieBasis_depth(PyLieBasis *basis);

RPY_NO_EXPORT
npy_intp PyLieBasis_size(PyLieBasis *basis);

RPY_NO_EXPORT
npy_intp PyLieBasis_true_size(PyLieBasis *basis);

RPY_NO_EXPORT
PyArrayObject *PyLieBasis_degree_begin(PyLieBasis *basis);

RPY_NO_EXPORT
PyArrayObject *PyLieBasis_data(PyLieBasis *basis);

RPY_NO_EXPORT
npy_intp PyLieBasis_find_word(PyLieBasis* basis, const LieWord* target);

RPY_NO_EXPORT
int PyLieBasis_get_parents(PyLieBasis* basis, npy_intp index, LieWord* out);

RPY_NO_EXPORT
int32_t PyLieBasis_degree(PyLieBasis *basis, npy_intp key);

int init_lie_basis(PyObject *module);

#ifdef __cplusplus
}
#endif

#endif //ROUGHPY_COMPUTE__SRC_LIE_BASIS_H
