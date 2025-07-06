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
#include "xla/shape.h"
#include "xla/shape_util.h"

#include "shape.h"
#include "shape_util.h"

extern "C" {
    ShapeIndex* ShapeIndex_new() {
        return reinterpret_cast<ShapeIndex*>(new xla::ShapeIndex());
    }
    void ShapeIndex_delete(ShapeIndex* s) {
        delete reinterpret_cast<xla::ShapeIndex*>(s);
    }

    void ShapeIndex_push_back(ShapeIndex& shape_index, int value) {
        reinterpret_cast<xla::ShapeIndex&>(shape_index).push_back(value);
    }

    void ShapeIndex_push_front(ShapeIndex& shape_index, int value) {
        reinterpret_cast<xla::ShapeIndex&>(shape_index).push_front(value);
    }

    Shape* MakeShape(int primitive_type, int* shape, int rank) {
        int64_t shape64[rank];
        std::copy(shape, shape + rank, shape64);

        xla::Shape* xla_shape = new xla::Shape();
        *xla_shape = xla::ShapeUtil::MakeShape(
            (xla::PrimitiveType) primitive_type,
            absl::Span<const int64_t>(shape64, rank)
        );
        return reinterpret_cast<Shape*>(xla_shape);
    }
}
