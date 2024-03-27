//
// Created by sam on 3/26/24.
//

#include "arbitrary_precision_rational.h"
#include "scalar_serialization.h"


template <typename Integer>
void rpy::scalars::dtl::MPIntegerSerializationHelper<Integer>::save(
        cereal::JSONOutputArchive& archive,
        const std::uint32_t RPY_UNUSED_VAR version
) const
{
    RPY_SERIAL_SERIALIZE_NVP("is_negative", is_negative());
    RPY_SERIAL_SERIALIZE_SIZE(nbytes());
    archive.saveBinaryValue(limbs(), nbytes(), "data");
}

template <typename Integer>
void rpy::scalars::dtl::MPIntegerSerializationHelper<Integer>::load(
        cereal::JSONInputArchive& archive,
        const std::uint32_t RPY_UNUSED_VAR version
)
{
    bool is_negative;
    dimn_t size;

    RPY_SERIAL_SERIALIZE_VAL(is_negative);
    {
        // This is probably redundant, but keep the type system happy.
        serial::size_type tmp_size;
        RPY_SERIAL_SERIALIZE_SIZE(tmp_size);
        size = static_cast<dimn_t>(tmp_size);
    }

    if (size > 0) {
        auto n_limbs = (size + sizeof(limbs_t) - 1) / sizeof(limbs_t);
        archive.loadBinaryValue(resize(n_limbs), size, "data");
        finalize(n_limbs, is_negative);
    }
}

template <typename Integer>
void rpy::scalars::dtl::MPIntegerSerializationHelper<Integer>::save(
        cereal::XMLOutputArchive& archive,
        const std::uint32_t RPY_UNUSED_VAR version
) const
{
    RPY_SERIAL_SERIALIZE_NVP("is_negative", is_negative());
    RPY_SERIAL_SERIALIZE_SIZE(nbytes());
    archive.saveBinaryValue(limbs(), nbytes(), "data");
}

template <typename Integer>
void rpy::scalars::dtl::MPIntegerSerializationHelper<Integer>::load(
        cereal::XMLInputArchive& archive,
        const std::uint32_t RPY_UNUSED_VAR version
)
{
    bool is_negative;
    dimn_t size;

    RPY_SERIAL_SERIALIZE_VAL(is_negative);

    {
        // This is probably redundant, but keep the type system happy.
        serial::size_type tmp_size;
        RPY_SERIAL_SERIALIZE_SIZE(tmp_size);
        size = static_cast<dimn_t>(tmp_size);
    }
    if (size > 0) {
        auto n_limbs = (size + sizeof(limbs_t) - 1) / sizeof(limbs_t);
        archive.loadBinaryValue(resize(n_limbs), size, "data");
        finalize(n_limbs, is_negative);
    }
}
