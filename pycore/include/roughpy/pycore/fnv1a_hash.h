#ifndef ROUGHPY_PYCORE_FNV1A_HASH_H
#define ROUGHPY_PYCORE_FNV1A_HASH_H


#include "py_headers.h"
#include <roughpy/core/fnv1a_hash.h>

#if SIZEOF_PY_HASH_T == 8
#  define FNV1A_OFFSET_BASIS FNV1A_OFFSET_BASIS_64
#elif SIZEOF_PY_HASH_T == 4
#  define FNV1A_OFFSET_BASIS FNV1A_OFFSET_BASIS_32
#else
#  error "Only 32 bit or 64 bit hash values are supported!"
#endif

#ifdef __cplusplus
extern "C" {
#endif

    static inline Py_uhash_t fnv1a_hash_string(Py_uhash_t state, const char* str)
    {
#if SIZEOF_VOID_P == 8
        return fnv1a_hash_string64(state, str);
#else
        return fnv1a_hash_string32(state, str);
#endif
    }

    static inline Py_uhash_t
    fnv1a_hash_bytes(Py_uhash_t state, const void* bytes, size_t len)
    {
#if SIZEOF_VOID_P == 8
        return fnv1a_hash_bytes64(state, bytes, len);
#else
        return fnv1a_hash_bytes32(state, bytes, len);
#endif
    }

    static inline Py_uhash_t fnv1a_hash_i32(Py_uhash_t state, int32_t value)
    {
        return fnv1a_hash_bytes(state, &value, sizeof(value));
    }

    static inline Py_uhash_t fnv1a_hash_u32(Py_uhash_t state, uint32_t value)
    {
        return fnv1a_hash_bytes(state, &value, sizeof(value));
    }

    static inline Py_uhash_t fnv1a_hash_i64(Py_uhash_t state, int64_t value)
    {
        return fnv1a_hash_bytes(state, &value, sizeof(value));
    }

    static inline Py_uhash_t fnv1a_hash_u64(Py_uhash_t state, uint64_t value)
    {
        return fnv1a_hash_bytes(state, &value, sizeof(value));
    }

    static inline Py_uhash_t fnv1a_hash_isize(Py_uhash_t state, Py_ssize_t value)
    {
        return fnv1a_hash_bytes(state, &value, sizeof(value));
    }

    static inline Py_uhash_t fnv1a_hash_usize(Py_uhash_t state, size_t value)
    {
        return fnv1a_hash_bytes(state, &value, sizeof(value));
    }

    static inline Py_uhash_t fnv1a_hash_ptr(Py_uhash_t state, const void* value)
    {
        return fnv1a_hash_bytes(state, &value, sizeof(value));
    }

    static inline Py_hash_t fnv1a_finalize_hash(Py_hash_t state)
    {
        Py_hash_t hash = (Py_hash_t) state;

        if (hash == -1) { hash = -2; }

        return hash;
    }


#ifdef __cplusplus
}
#endif

#endif// ROUGHPY_PYCORE_FNV1A_HASH_H
