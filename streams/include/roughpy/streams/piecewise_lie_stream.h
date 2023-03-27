#ifndef ROUGHPY_STREAMS_PIECEWISE_LIE_STREAM_H_
#define ROUGHPY_STREAMS_PIECEWISE_LIE_STREAM_H_


#include "stream_base.h"

#include <roughpy/core/implementation_types.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/algebra/context.h>

namespace rpy { namespace streams {


class PiecewiseLieStream : public StreamInterface {
public:
    using LiePiece = std::pair<intervals::RealInterval, algebra::Lie>;

private:

    std::vector<LiePiece> m_data;

    static inline scalars::Scalar to_multiplier_upper(const intervals::RealInterval& interval, param_t param) {
        assert(interval.inf() <= param && param <= interval.sup());
        return scalars::Scalar((interval.sup() - param) / (interval.sup() - interval.inf()));
    }
    static inline scalars::Scalar to_multiplier_lower(const intervals::RealInterval& interval, param_t param) {
        assert(interval.inf() <= param && param <= interval.sup());
        return scalars::Scalar((param - interval.inf()) / (interval.sup() - interval.inf()));
    }

public:
    PiecewiseLieStream(std::vector<LiePiece>&& arg, StreamMetadata&& md);


    bool empty(const intervals::Interval& interval) const noexcept override;

protected:
    algebra::Lie log_signature_impl(const intervals::Interval& domain, const algebra::Context& ctx) const override;

};


}}


#endif // ROUGHPY_STREAMS_PIECEWISE_LIE_STREAM_H_
