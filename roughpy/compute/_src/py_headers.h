#ifndef ROUGHPY_COMPUTE__SRC_PY_HEADERS_H
#define ROUGHPY_COMPUTE__SRC_PY_HEADERS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>


#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#ifndef NPY_IMPORT_THE_APIS_PLEASE
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#endif
#define PY_ARRAY_UNIQUE_SYMBOL RPY_COMPUTE_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL RPY_COMPUTE_UFUNC_API

#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>


#define PYVER_HEX(major, minor) \
     (((major) << 24) | \
      ((minor) << 16))



#if defined(__GNUC__) || defined(__clang__)
#define RPY_NO_EXPORT __attribute__((visibility("hidden")))
#else
#define RPY_NO_EXPORT
#endif


#define RPY_CPT_TYPE_NAME(type) "_rpy_compute_internals." # type

#define RPY_STATUS_OR_RETURN(status, ret)                                      \
    do {                                                                       \
        int code = (status);                                                   \
        if (code < 0) { return (ret); }                                        \
    } while (0)

#define RPY_OBJ_OR_RETURN(obj, ret)                                            \
    do {                                                                       \
        if ((obj) == NULL) { return (ret); }                                   \
    } while (0)

#define RPY_STATUS_OR_RETURN_NULL(status) RPY_STATUS_OR_RETURN(status, NULL)
#define RPY_STATUS_OR_RETURN_INT(status) RPY_STATUS_OR_RETURN(status, -1)
#define RPY_OBJ_OR_RETURN_NULL(obj) RPY_OBJ_OR_RETURN(obj, NULL)
#define RPY_OBJ_OR_RETURN_INT(obj) RPY_OBJ_OR_RETURN(obj, -1)

#endif //ROUGHPY_COMPUTE__SRC_PY_HEADERS_H
