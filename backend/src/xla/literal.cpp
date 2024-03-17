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
#include "xla/literal.h"

#include "literal.h"
#include "shape.h"
#include "shape_util.h"

extern "C" {
    // technically for LiteralBase. Do we care?
    void* Literal_untyped_data(Literal* s) {
        std::cout << "Literal_untyped_data ..." << std::endl;
        return reinterpret_cast<xla::Literal*>(s)->untyped_data();
    }

    int64_t Literal_size_bytes(Literal* s) {
        std::cout << "Literal_size_bytes ..." << std::endl;
        return reinterpret_cast<xla::Literal*>(s)->size_bytes();
    }

    Literal* Literal_new(Shape& shape) {
        xla::Shape& shape_ = reinterpret_cast<xla::Shape&>(shape);
        xla::Literal* lit = new xla::Literal(shape_, true);
        return reinterpret_cast<Literal*>(lit);
    }

    void Literal_delete(Literal* lit) {
        delete reinterpret_cast<xla::Literal*>(lit);
    }
}

template <typename NativeT>
NativeT Literal_Get(Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index) {
    xla::Literal& lit_ = reinterpret_cast<xla::Literal&>(lit);
    int64_t multi_index_[multi_index_len];
    std::copy(multi_index, multi_index + multi_index_len, multi_index_);
    auto multi_index_span = absl::Span<const int64_t>(multi_index_, multi_index_len);
    auto& shape_index_ = reinterpret_cast<xla::ShapeIndex&>(shape_index);
    return lit_.Get<NativeT>(multi_index_span, shape_index_);
};

template <typename NativeT>
void Literal_Set(
    Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index, NativeT value
) {
    xla::Literal& lit_ = reinterpret_cast<xla::Literal&>(lit);
    int64_t multi_index_[multi_index_len];
    std::copy(multi_index, multi_index + multi_index_len, multi_index_);
    auto multi_index_span = absl::Span<const int64_t>(multi_index_, multi_index_len);
    auto& shape_index_ = reinterpret_cast<xla::ShapeIndex&>(shape_index);
    lit_.Set<NativeT>(multi_index_span, shape_index_, value);
};

extern "C" {
    int Literal_Get_bool(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index
    ) {
        return (int) Literal_Get<bool>(lit, multi_index, multi_index_len, shape_index);
    }

    int Literal_Get_int32_t(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index
    ) {
        return Literal_Get<int32_t>(lit, multi_index, multi_index_len, shape_index);
    }

    int Literal_Get_uint32_t(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index
    ) {
        return (int) Literal_Get<uint32_t>(lit, multi_index, multi_index_len, shape_index);
    }

    int Literal_Get_uint64_t(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index
    ) {
        return (int) Literal_Get<uint64_t>(lit, multi_index, multi_index_len, shape_index);
    }

    double Literal_Get_double(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index
    ) {
        return Literal_Get<double>(lit, multi_index, multi_index_len, shape_index);
    }

    void Literal_Set_bool(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index, int value
    ) {
        Literal_Set<bool>(lit, multi_index, multi_index_len, shape_index, (bool) value);
    }

    void Literal_Set_int32_t(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index, int value
    ) {
        Literal_Set<int32_t>(lit, multi_index, multi_index_len, shape_index, value);
    }

    void Literal_Set_uint32_t(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index, int value
    ) {
        Literal_Set<uint32_t>(lit, multi_index, multi_index_len, shape_index, (uint32_t) value);
    }

    void Literal_Set_uint64_t(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index, int value
    ) {
        Literal_Set<uint64_t>(lit, multi_index, multi_index_len, shape_index, (uint64_t) value);
    }

    void Literal_Set_double(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index, double value
    ) {
        Literal_Set<double>(lit, multi_index, multi_index_len, shape_index, value);
    }
}
