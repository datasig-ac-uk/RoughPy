#ifndef ROUGHPY_COMPUTE__SRC_LIE_MULTIPLICATION_CACHE_H
#define ROUGHPY_COMPUTE__SRC_LIE_MULTIPLICATION_CACHE_H

#include "py_headers.h"
#include "lie_basis.h"

#ifdef __cplusplus
extern "C" {
#endif



typedef struct {
    npy_intp word[2];
    npy_intp size;
    npy_intp data[1];
} LieMultiplicationCacheEntry;

struct PyLieMultiplicationCacheInner;

typedef struct PyLieMultiplicationCache {
    PyObject_HEAD
    struct PyLieMultiplicationCacheInner* inner;
} PyLieMultiplicationCache;


RPY_NO_EXPORT
PyObject* get_lie_multiplication_cache(PyObject* basis);


RPY_NO_EXPORT
const LieMultiplicationCacheEntry* PyLieMultiplicationCache_get(PyLieMultiplicationCache* cache, PyObject* basis_ob, const LieWord* word);


RPY_NO_EXPORT
PyObject* lie_mutiplication_cache_clear(PyObject* cache);

RPY_NO_EXPORT
PyLieMultiplicationCacheInner* lie_multiplication_cache_to_inner(PyObject* cache);


RPY_NO_EXPORT
int init_lie_multiplication_cache(PyObject* module);


#ifdef __cplusplus
}
#endif

#endif //ROUGHPY_COMPUTE__SRC_LIE_MULTIPLICATION_CACHE_H