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

    Shape* MakeTupleShape(Shape* shapes, size_t shapes_len) {
        auto shapes_ = reinterpret_cast<xla::Shape*>(shapes);
        auto shapes_span = absl::Span<const xla::Shape>(shapes_, shapes_len);

        auto shape = xla::ShapeUtil::MakeTupleShape(shapes_span);

        return reinterpret_cast<Shape*>(new xla::Shape(shape));
    }

    Shape* MakeShape(int primitive_type, int* shape, size_t rank) {
        int64_t shape64[rank];
        std::copy(shape, shape + rank, shape64);

        auto xla_shape = xla::ShapeUtil::MakeShape(
            (xla::PrimitiveType) primitive_type,
            absl::Span<const int64_t>(shape64, rank)
        );

        return reinterpret_cast<Shape*>(new xla::Shape(xla_shape));
    }

//
//    Shape* MakeTupleShape(Shape* shapes, size_t shapes_len) {
//        auto shapes_ = reinterpret_cast<xla::Shape*>(shapes);
//
////        for (int i = 0; i < shapes_len; i++) {
////          printf("%d: ", i);
////          printf("%s\n", xla::ShapeUtil::HumanString(shapes_[i]).c_str());
////        }
//        std::vector<xla::Shape> tuple_shapes(shapes_, shapes_ + shapes_len);
//
//        auto xla_shape = new xla::Shape(tuple_shapes);
//
////        printf("tuple: %s\n", xla::ShapeUtil::HumanString(*xla_shape).c_str());
//
//        return reinterpret_cast<Shape*>(xla_shape);
//    }
//
//    Shape* MakeShape(int primitive_type, int* shape, size_t rank) {
//        int64_t shape64[rank];
//        std::copy(shape, shape + rank, shape64);
//
//        auto xla_shape = new xla::Shape(
//            (xla::PrimitiveType) primitive_type,
//            absl::Span<const int64_t>(shape64, rank)
//        );
//
//        return reinterpret_cast<Shape*>(xla_shape);
//    }

}
