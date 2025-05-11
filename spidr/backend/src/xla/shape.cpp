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

#include "shape.h"
#include "xla/shape_util.h"

extern "C" {
    void Shape_delete(Shape* s) {
        delete reinterpret_cast<xla::Shape*>(s);
    }

    size_t sizeof_Shape() {
        return sizeof(xla::Shape);
    }

    Shape* mallocShapeArray(size_t size) {
        return reinterpret_cast<Shape*>(new xla::Shape[size]);
    }

    void set_array_Shape(Shape* arr, size_t idx, Shape* shape) {
        auto arr_ = reinterpret_cast<xla::Shape*>(arr);
        auto shape_ = reinterpret_cast<xla::Shape*>(shape);
        // ideas:
        // * array has not enough space allocated, and so is being overwritten
        // * something is being freed too early

//        printf("set_array_Shape\n");
//        printf("%zu: ", idx);
//        printf("%s\n", xla::ShapeUtil::HumanString(*shape_).c_str());

//        printf("get shape_\n");
//        printf("sizeof shape_: %d\n", (int) sizeof(shape_));
//        printf("assign\n");
        arr_[idx] = *shape_;

//        printf("set_array_Shape return\n");
    }

//    // this seems to work
//    Shape* shapearray(Shape* shape) {
//        auto shape_ = reinterpret_cast<xla::Shape*>(shape);
//        // ideas:
//        // * array has not enough space allocated, and so is being overwritten
//        // * something is being freed too early
//        auto arr_ = new xla::Shape[1];
//        arr_[0] = *shape_;
//
//        return reinterpret_cast<Shape*>(arr_);
//    }
}
