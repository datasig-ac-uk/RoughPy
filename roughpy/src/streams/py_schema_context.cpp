//
// Created by user on 04/07/23.
//

#include "py_schema_context.h"

#include "intervals/date_time_interval.h"

using namespace rpy;
using namespace streams;

intervals::RealInterval python::PySchemaContext::convert_parameter_interval(
        const intervals::Interval& arg) const
{
    const auto* dt_interval_ptr
            = dynamic_cast<const python::DateTimeInterval*>(&arg);
    if (dt_interval_ptr != nullptr && m_dt_conversion) {
        const auto& options = *m_dt_conversion;

        auto inf = python::convert_delta_from_datetimes(
                dt_interval_ptr->dt_inf(), m_dt_reference, options);
        auto sup = python::convert_delta_from_datetimes(
                dt_interval_ptr->dt_sup(), m_dt_reference, options);

        return reparametrize({inf, sup, arg.type()});
    }
    return SchemaContext::convert_parameter_interval(arg);
}
