#ifndef RPY_PY_SCALARS_SCALARS_H_
#define RPY_PY_SCALARS_SCALARS_H_

#include "roughpy_module.h"

#include <functional>


#include <roughpy/scalars/scalar_pointer.h>
#include <roughpy/scalars/scalar_type.h>
#include <roughpy/scalars/key_scalar_array.h>

namespace rpy {
namespace __attribute__((visibility("hidden"))) python {


struct AlternativeKeyType {
    py::handle py_key_type;
    std::function<key_type(py::handle)> converter;
};

struct PyToBufferOptions {
    /// Scalar type to use. If null, will be set to the resulting type
    const scalars::ScalarType *type = nullptr;

    /// Maximum number of nested objects to search. Set to 0 for no recursion.
    dimn_t max_nested = 0;

    /// Information about the constructed array
    std::vector<idimn_t> shape;

    /// Allow a single, untagged scalar as argument
    bool allow_scalar = true;

    /// Do not check std library types or imported data types.
    /// All Python types will (try) to be converted to double.
    bool no_check_imported = false;

    /// cleanup function to be called when we're finished with the data
    std::function<void()> cleanup = nullptr;

    /// Alternative acceptable key_type/conversion pair
    AlternativeKeyType *alternative_key = nullptr;
};

scalars::KeyScalarArray py_to_buffer(const py::handle &arg, PyToBufferOptions &options);

void assign_py_object_to_scalar(scalars::ScalarPointer ptr, py::handle object);

void init_scalars(py::module_& m);

} // namespace python
} // namespace rpy

#endif // RPY_PY_SCALARS_SCALARS_H_
