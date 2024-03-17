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
#include <iostream>

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

    Shape* MakeTupleShape(Shape* shapes, int shapes_len) {
        std::cout << "MakeTupleShape ..." << std::endl;
        auto shapes_ = reinterpret_cast<xla::Shape*>(shapes);
        auto shapes_span = absl::Span<xla::Shape>(shapes_, shapes_len);

        xla::Shape* xla_shape = new xla::Shape();
        *xla_shape = xla::ShapeUtil::MakeTupleShape(shapes_span);
        return reinterpret_cast<Shape*>(xla_shape);
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
