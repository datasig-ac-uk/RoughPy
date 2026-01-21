#ifndef ROUGHPY_COMPUTE__SRC_CHECK_DIMS_HPP
#define ROUGHPY_COMPUTE__SRC_CHECK_DIMS_HPP

#include <roughpy/pycore/py_headers.h>

namespace rpy::compute {

[[gnu::always_inline]] inline
bool check_dims(npy_intp const* shape1,
                npy_intp ndims1,
                npy_intp const* shape2,
                npy_intp ndims2)
{
    if (ndims1 != ndims2) { return false; }

    for (npy_intp i = 0; i < ndims1; ++i) {
        if (shape1[i] != shape2[i]) { return false; }
    }

    return true;
}

} // namespace rpy::compute

#endif //ROUGHPY_COMPUTE__SRC_CHECK_DIMS_HPP
