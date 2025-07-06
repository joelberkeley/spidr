/*
Copyright (C) 2022  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#include "xla/literal.h"

#include "literal.h"
#include "shape.h"
#include "shape_util.h"

extern "C" {
    void* Literal_untyped_data(Literal* s) {
        return reinterpret_cast<xla::Literal*>(s)->untyped_data();
    }

    int64_t Literal_size_bytes(Literal* s) {
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
