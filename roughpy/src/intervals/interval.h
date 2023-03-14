#ifndef RPY_PY_INTERVALS_INTERVAL_H_
#define RPY_PY_INTERVALS_INTERVAL_H_

#include "roughpy_module.h"

#include <roughpy/intervals/interval.h>

namespace rpy {
namespace python {

class PyInterval : intervals::Interval {
public:
    using intervals::Interval::Interval;

private:
    param_t inf() const override;
    param_t sup() const override;

    param_t included_end() const override;
    param_t excluded_end() const override;
    bool contains(param_t arg) const noexcept override;
    bool is_associated(const Interval &arg) const noexcept override;
    bool contains(const Interval &arg) const noexcept override;
    bool intersects_with(const Interval &arg) const noexcept override;
};


void init_interval(py::module_& m);

} // namespace python
} // namespace rpy

#endif // RPY_PY_INTERVALS_INTERVAL_H_
