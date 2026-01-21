#ifndef ROUGHPY_COMPUTE__SRC_LIE_BASIS_H
#define ROUGHPY_COMPUTE__SRC_LIE_BASIS_H

#include <roughpy/pycore/py_headers.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * There are two definitions of a Hall set. The common definition is
 * as follows:
 *
 * Let $H$ be a subset of $M(A)$ (the free magma on $A$). The common elements of
 * the definitions from Bourbaki (Lie Groups and Lie Algebras 1-3, II.2.10) and
 * Reutenauer (Chapter 4) are as follows:
 *
 *  1) H has a total order $\leq$.
 *  2) A is contained in H.
 *
 * The definitions then differ slightly. The remaining parts of the definition
 * from Bourbaki are as follows:
 *
 *  3) Every element of $H\cap M^2(A)$ is of the form $(x, y)$ with $x < y$.
 *     (Here $M^2(A)$ denotes the elements of $M(A)$ of length 2.)
 *  4) An element $w\in M(A)$ of length $\geq 3$ belongs to $H$ if and only if
 *     it is of the form $w = (a, (b, c))$ with $a, b, c\in H$, $(b, c) \in H$,
 *     and $b\leq a < (b, c)$ and $b < c$.
 *
 * Bourbaki also places a condition on $\leq$ whereby if $u, v\in H$ with
 * $l(u) < l(v)$ then $u < v$. (Here $l(u)$ denotes the length of $u$.) This
 * condition essentially places the elements of $H$ in degree-order. We impose
 * this condition too, because it is necessary for the degree_begin array to
 * have meaning, though we might change what we mean by "length".
 *
 * The definition from Reutenauer is as follows.
 *
 *  3) For any $h = (h', h'') \in H\setminus A$ one has $h''\in H$ and
 *     $h < h''$.
 *  4) For any tree $w = (h', h'') \in M(A)\setminus A$ one has $w\in H$ if and
 *     only if $h', h''\in H$ and $h' < h''$ and either $h'\in A$ or $h'=(x, y)$
 *     with $h'' \leq y$.
 *
 * The most consequential difference here is the condition is the factor of $w$
 * on which the ordering condition is applied. In Bourbaki, the right factor
 * $(b, c)$ has the condition $b\leq a < (b, c)$. In Reutenauer, it is the
 * left factor $h' = (x, y)$ on which the condition is applied with
 * $h' < h'' \leq y$.
 *
 * Now the standard Hall sets that we construct uses the Bourbaki definition,
 * but one might reasonably expect users to provide Hall sets that follow the
 * Reutenauer definition. To support this, we add some constants that dictate
 * which version of the definition one should use. This will have consequences
 * for how the basis should be used when, for instance, computing brackets,
 * or when checking if basis data is valid.
 *
 * For now, this is only used in the check function.
 */
typedef enum _PyLieBasis_Major
{
    PLB_Major_Bourbaki = 0,
    PLB_Major_Right = PLB_Major_Bourbaki,
    PLB_Major_Reutenauer = 1,
    PLB_Major_Left = PLB_Major_Reutenauer
} PyLieBasis_Major;

typedef union _LieWord
{
    npy_intp letters[2];
    struct {
        npy_intp left;
        npy_intp right;
    };
} LieWord;

typedef struct _PyLieBasis PyLieBasis;

extern PyTypeObject* PyLieBasis_Type;

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

PyObject* get_l2t_matrix(PyObject* basis, PyObject* dtype_obj);

PyObject* get_t2l_matrix(PyObject* basis, PyObject* dtype_obj);

static inline int PyLieBasis_Check(PyObject* obj)
{
    return PyObject_TypeCheck(obj, PyLieBasis_Type);
}

RPY_NO_EXPORT
int32_t PyLieBasis_width(PyLieBasis* basis);

RPY_NO_EXPORT
int32_t PyLieBasis_depth(PyLieBasis* basis);

RPY_NO_EXPORT
npy_intp PyLieBasis_size(PyLieBasis* basis);

RPY_NO_EXPORT
npy_intp PyLieBasis_true_size(PyLieBasis* basis);

RPY_NO_EXPORT
PyArrayObject* PyLieBasis_degree_begin(PyLieBasis* basis);

RPY_NO_EXPORT
PyArrayObject* PyLieBasis_data(PyLieBasis* basis);

RPY_NO_EXPORT
npy_intp PyLieBasis_find_word(
        PyLieBasis* basis,
        const LieWord* target,
        int32_t degree_hint
);

RPY_NO_EXPORT
int PyLieBasis_get_parents(PyLieBasis* basis, npy_intp index, LieWord* out);

RPY_NO_EXPORT
int32_t PyLieBasis_degree(PyLieBasis* basis, npy_intp key);

PyObject* PyLieBasis_key2str(PyLieBasis* basis, npy_intp key);

PyObject* PyLieBasis_word2str(PyLieBasis* basis, const LieWord* word);

PyObject* PyLieBasis_get(int32_t width, int32_t depth);

int PyLieBasis_check_data_internal(
        PyArrayObject* data,
        PyArrayObject* degree_begin,
        int32_t width,
        int32_t depth,
        PyObject* total_order,
        PyLieBasis_Major major,
        char const** message
);


npy_intp compute_lie_degree_dim(int32_t width, int32_t degree);


npy_intp compute_lie_dim(const int32_t width, const int32_t depth);


PyObject*
PyLieBasis_check_data(PyObject* self, PyObject* args, PyObject* kwargs);

int init_lie_basis(PyObject* module);

#ifdef __cplusplus
}
#endif

#endif// ROUGHPY_COMPUTE__SRC_LIE_BASIS_H
