//
// Created by sam on 24/06/24.
//

#ifndef ROUGHPY_SCALARS_SCALAR_STREAM_H
#define ROUGHPY_SCALARS_SCALAR_STREAM_H


#include "scalars_fwd.h"

#include <roughpy/core/container/vector.h>

#include "scalar_array.h"

namespace rpy { namespace scalars {


class ROUGHPY_SCALARS_EXPORT ScalarStream
{
protected:
    containers::Vec<ScalarArray> m_stream;
    TypePtr p_type;

public:
    RPY_NO_DISCARD TypePtr type() const noexcept { return p_type; }

    ScalarStream();
    ScalarStream(const ScalarStream& other);
    ScalarStream(ScalarStream&& other) noexcept;

    explicit ScalarStream(TypePtr type);
    ScalarStream(ScalarArray base, containers::Vec<dimn_t> shape);

    ScalarStream(
            containers::Vec<ScalarArray>&& stream,
            dimn_t row_elts,
            TypePtr type
    )
        : m_stream(std::move(stream)),
          p_type(type)
    {}

    ScalarStream& operator=(const ScalarStream& other);
    ScalarStream& operator=(ScalarStream&& other) noexcept;

    RPY_NO_DISCARD dimn_t col_count(dimn_t i = 0) const noexcept;
    RPY_NO_DISCARD dimn_t row_count() const noexcept { return m_stream.size(); }

    RPY_NO_DISCARD dimn_t max_row_size() const noexcept;

    RPY_NO_DISCARD ScalarArray operator[](dimn_t row) const noexcept;
    RPY_NO_DISCARD ScalarCRef operator[](std::pair<dimn_t, dimn_t> index
    ) const noexcept;

    void set_ctype(TypePtr type) noexcept;

    void reserve_size(dimn_t num_rows);

    void push_back(const ScalarArray& data);
    void push_back(ScalarArray&& data);
};

}}

#endif //ROUGHPY_SCALARS_SCALAR_STREAM_H
