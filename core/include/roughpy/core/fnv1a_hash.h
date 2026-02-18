#ifndef ROUGHPY_CORE_FNV1A_HASH_H
#define ROUGHPY_CORE_FNV1A_HASH_H

#include <stdint.h>
#include <stddef.h>

/*
 * This is an implementation of the FNV-1a hash function which essentially
 * follows the implementation here: https://github.com/lcn2/fnv. This is the
 * same hash function that is used in NumPy (and is referenced is implemented
 * in the Python source code).
 */

#define FNV1A_OFFSET_BASIS_64 0xCBF29CE484222325
#define FNV1A_PRIME_64 0x00000100000001B3
#define FNV1A_OFFSET_BASIS_32 0x811C9DC5
#define FNV1A_PRIME_32 0x01000193


#ifdef __cplusplus
extern "C" {
#endif

uint32_t
fnv1a_hash_string32(uint32_t state, const char* str);
uint64_t
fnv1a_hash_string64(uint64_t state, const char* str);

uint32_t
fnv1a_hash_bytes32(uint32_t state, const void* bytes, size_t len);
uint64_t
fnv1a_hash_bytes64(uint64_t state, const void* bytes, size_t len);


#ifdef __cplusplus
}
#endif

#endif// ROUGHPY_CORE_FNV1A_HASH_H
