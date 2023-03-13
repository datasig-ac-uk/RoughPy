#ifndef ROUGHPY_SCALARS_SCALAR_STREAM_H_
#define ROUGHPY_SCALARS_SCALAR_STREAM_H_

#include "scalars_fwd.h"

#include <vector>

namespace rpy { namespace scalars {

class ROUGHPY_SCALARS_EXPORT ScalarStream {
    std::vector<const void *> m_stream;
    //    boost::container::small_vector<dimn_t, 1> m_elts_per_row;
    std::vector<dimn_t> m_elts_per_row;
    const ScalarType *p_type;

public:
    const ScalarType *type() const noexcept { return p_type; }

    ScalarStream();

    explicit ScalarStream(const ScalarType *type);
    ScalarStream(ScalarPointer base, std::vector<dimn_t> shape);

    ScalarStream(std::vector<const void *> &&stream, dimn_t row_elts, const ScalarType *type)
        : m_stream(stream), m_elts_per_row{row_elts}, p_type(type) {}

    dimn_t col_count(dimn_t i = 0) const noexcept;
    dimn_t row_count() const noexcept { return m_stream.size(); }

    ScalarArray operator[](dimn_t row) const noexcept;
    Scalar operator[](std::pair<dimn_t, dimn_t> index) const noexcept;

    void set_elts_per_row(dimn_t num_elts) noexcept;
    void set_ctype(const scalars::ScalarType *type) noexcept;

    void reserve_size(dimn_t num_rows);

    void push_back(const ScalarPointer &data);
    void push_back(const ScalarArray &data);
};

}}

#endif // ROUGHPY_SCALARS_SCALAR_STREAM_H_
