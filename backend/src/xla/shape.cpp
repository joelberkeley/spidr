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

#include "shape.h"

extern "C" {
    void Shape_delete(Shape* s) {
        delete reinterpret_cast<xla::Shape*>(s);
    }

    int sizeof_Shape() {
//        std::cout << "sizeof_Shape ..." << std::endl;
        return sizeof(xla::Shape);
    }

    void set_array_Shape(Shape* arr, int idx, Shape* shape) {
//        std::cout << "set_array_Shape ..." << std::endl;
//        std::cout << "... shape " << shape << std::endl;
        reinterpret_cast<xla::Shape*>(arr)[idx] = *reinterpret_cast<xla::Shape*>(shape);
    }
}
