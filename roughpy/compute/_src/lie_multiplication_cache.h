#ifndef ROUGHPY_COMPUTE__SRC_LIE_MULTIPLICATION_CACHE_H
#define ROUGHPY_COMPUTE__SRC_LIE_MULTIPLICATION_CACHE_H

#include "py_headers.h"
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

/**
 * Creates a new instance of PyLieMultiplicationCache with the specified width.
 *
 * This function allocates and initializes a new PyLieMultiplicationCache
 * object. It sets up the internal cache structure and ensures memory
 * management is handled appropriately. The width parameter defines the
 * maximum allowable size or dimensionality for this cache.
 *
 * The function manages resource allocation failures by cleaning up and
 * returning appropriate error signals.
 *
 * @param width The dimensional width or size constraint for the Lie
 * multiplication cache to be created.
 *
 * @return A pointer to the newly created PyLieMultiplicationCache object if
 *         successful, or nullptr if the creation fails due to memory allocation
 *         issues or other unforeseen errors.
 */
RPY_NO_EXPORT
PyObject* PyLieMultiplicationCache_new(int32_t width);

/**
 * Retrieves the Lie multiplication cache associated with the given Lie basis.
 *
 * This function retrieves or creates a cached instance of the Lie
 * multiplication cache for the specified PyLieBasis object. If the cache for
 * the given basis width does not exist, a new instance is created, initialized,
 * and stored in the internal cache structure. The function ensures proper
 * memory management and reference counting during retrieval or creation.
 *
 * @param basis A pointer to a PyLieBasis object representing the basis of the
 * Lie algebra for which the multiplication cache is requested.
 *
 * @return A PyObject representing the cached Lie multiplication instance if
 * successfully retrieved or created. If an error occurs, such as memory
 * allocation failure or invalid input, returns nullptr.
 */
RPY_NO_EXPORT
PyObject* get_lie_multiplication_cache(PyLieBasis* basis);

/**
 * Retrieves or computes the multiplication result for a given Lie word from
 * the cache.
 *
 * This function attempts to locate the result of multiplying the given Lie
 * word in the cache associated with the provided PyLieMultiplicationCache.
 * If the result is not already cached, it computes the result, populates the
 * cache, and ensures the result is represented in the required canonical form.
 *
 * The function performs several checks and validations:
 * - Verifies compatibility between the basis and cache.
 * - Validates the input word and confirms the degree does not exceed the
 *   basis depth.
 * - Checks and enforces the canonical representation of the word.
 * - Computes the multiplication result if the required data is not already
 *   cached, including accounting for potential sign adjustments when swapping
 *   terms to canonicalize the word.
 *
 * @param cache The Lie multiplication cache to retrieve or store results.
 * @param basis The corresponding Lie basis for the given word.
 * @param word The Lie word for which the multiplication result is requested.
 * @param degree The target degree for the resulting computation, used for
 *               validation screening. Set to -1 if the degree of the product
 *               is not yet known.
 *
 * @returns A pointer to the cached or newly computed Lie multiplication
 *          result entry, or nullptr if the computation failed.
 */
const LieMultiplicationCacheEntry*
PyLieMultiplicationCache_get(PyLieMultiplicationCache* cache,
    PyLieBasis* basis, const LieWord* word, int32_t degree);

/**
 * Clears the contents of the Lie multiplication cache.
 *
 * This function removes all entries from the cache associated with the given
 * PyLieMultiplicationCache object. It ensures that the internal cache structure
 * is emptied and ready to be reused without residual data. The function checks
 * for valid cache types before performing the operation and raises an error if
 * the provided object is not a valid Lie multiplication cache.
 *
 * @param cache A pointer to the PyLieMultiplicationCache object to be cleared.
 *
 * @return Py_None on successful clearing of the cache, or nullptr if the
 *         provided object is not compatible with the PyLieMultiplicationCache
 *         type.
 */
RPY_NO_EXPORT
PyObject* lie_multiplication_cache_clear(PyObject* cache);

/**
 * Initializes the Lie multiplication cache for the given Python module.
 *
 * This function sets up the necessary components for Lie multiplication
 * caching within the specified Python module. It initializes the
 * PyLieMultiplicationCache type, allocates memory for the cache, and adds
 * the cache as an object to the provided module. It ensures proper error
 * handling and resource cleanup in case of initialization failures.
 *
 * @param module The Python module where the Lie multiplication cache will
 *        be initialized and registered.
 *
 * @return 0 if the initialization is successful, or -1 if an error occurs
 *         during the initialization process, such as memory allocation
 *         failure or module registration issues.
 */
RPY_NO_EXPORT
int init_lie_multiplication_cache(PyObject* module);


#ifdef __cplusplus
}
#endif

#endif //ROUGHPY_COMPUTE__SRC_LIE_MULTIPLICATION_CACHE_H