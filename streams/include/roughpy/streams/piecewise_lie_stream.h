#ifndef ROUGHPY_STREAMS_PIECEWISE_LIE_STREAM_H_
#define ROUGHPY_STREAMS_PIECEWISE_LIE_STREAM_H_


#include "stream_base.h"

#include <roughpy/config/implementation_types.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/algebra/context.h>

namespace rpy { namespace streams {


class PiecewiseLiePath : public StreamInterface {
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

    PiecewiseLiePath(std::vector<LiePiece>&& arg, StreamMetadata&& md);

    using StreamInterface::log_signature;

    bool empty(const intervals::Interval& interval) const noexcept override;

    algebra::Lie log_signature(const intervals::Interval& domain, const algebra::Context& ctx) const override;

};


}}


#endif // ROUGHPY_STREAMS_PIECEWISE_LIE_STREAM_H_
