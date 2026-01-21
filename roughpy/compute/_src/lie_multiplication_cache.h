#ifndef ROUGHPY_COMPUTE__SRC_LIE_MULTIPLICATION_CACHE_H
#define ROUGHPY_COMPUTE__SRC_LIE_MULTIPLICATION_CACHE_H

#include <roughpy/pycore/py_headers.h>
#include "lie_basis.h"

#ifdef __cplusplus
extern "C" {
#endif



typedef struct {
    LieWord word;
    npy_intp size;
    npy_intp data[2];
} LieMultiplicationCacheEntry;


typedef struct PyLieMultiplicationCache PyLieMultiplicationCache;

RPY_NO_EXPORT
PyObject* PyLieMultiplicationCache_new(int32_t width);

RPY_NO_EXPORT
PyObject* get_lie_multiplication_cache(PyLieBasis* basis);


const LieMultiplicationCacheEntry*
PyLieMultiplicationCache_get(PyLieMultiplicationCache* cache,
    PyLieBasis* basis, const LieWord* word);


RPY_NO_EXPORT
PyObject* lie_multiplication_cache_clear(PyObject* cache);



RPY_NO_EXPORT
int init_lie_multiplication_cache(PyObject* module);


#ifdef __cplusplus
}
#endif

#endif //ROUGHPY_COMPUTE__SRC_LIE_MULTIPLICATION_CACHE_H