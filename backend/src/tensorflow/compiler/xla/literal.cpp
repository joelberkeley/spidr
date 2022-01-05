/*
Copyright 2022 Joel Berkeley

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "tensorflow/compiler/xla/literal.h"

#include "literal.h"

extern "C" {
    struct Literal;

    Literal* Literal_new(int* shape, int rank, int primitive_type) {
        xla::int64 shape64[rank];
        std::copy(shape, shape + rank, shape64);

        const std::vector<bool> dynamic_dimensions(rank, false);

        xla::Shape xla_shape = xla::ShapeUtil::MakeShape(
            (xla::PrimitiveType) primitive_type,
            absl::Span<const xla::int64>(shape64, rank),
            dynamic_dimensions
        );

        xla::Literal* lit = new xla::Literal(xla_shape, true);
        return reinterpret_cast<Literal*>(lit);
    }

    void Literal_delete(Literal* lit) {
        delete reinterpret_cast<xla::Literal*>(lit);
    }
}

template <typename NativeT>
NativeT Literal_Get(Literal& lit, int* indices) {
    xla::Literal& lit_ = reinterpret_cast<xla::Literal&>(lit);
    xla::int64 rank = lit_.shape().rank();
    xla::int64 multi_index[rank];
    std::copy(indices, indices + rank, multi_index);
    return lit_.Get<NativeT>(absl::Span<const xla::int64>(multi_index, rank));
};

template <typename NativeT>
void Literal_Set(Literal& lit, int* indices, NativeT value) {
    xla::Literal& lit_ = reinterpret_cast<xla::Literal&>(lit);
    xla::int64 rank = lit_.shape().rank();
    xla::int64 multi_index[rank];
    std::copy(indices, indices + rank, multi_index);
    lit_.Set<NativeT>(absl::Span<const xla::int64>(multi_index, rank), value);
};

extern "C" {
    int Literal_Get_bool(Literal& lit, int* indices) {
        return (int) Literal_Get<bool>(lit, indices);
    }

    int Literal_Get_int(Literal& lit, int* indices) {
        return Literal_Get<int>(lit, indices);
    }

    double Literal_Get_double(Literal& lit, int* indices) {
        return Literal_Get<double>(lit, indices);
    }

    void Literal_Set_bool(Literal& lit, int* indices, int value) {
        Literal_Set<bool>(lit, indices, (bool) value);
    }

    void Literal_Set_int(Literal& lit, int* indices, int value) {
        Literal_Set<int>(lit, indices, value);
    }

    void Literal_Set_double(Literal& lit, int* indices, double value) {
        Literal_Set<double>(lit, indices, value);
    }
}
