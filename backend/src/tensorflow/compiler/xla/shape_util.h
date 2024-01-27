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
#include "shape.h"

extern "C" {
    struct ShapeIndex;

    ShapeIndex* ShapeIndex_new();
    void ShapeIndex_delete(ShapeIndex* s);
    void ShapeIndex_push_back(ShapeIndex& shape_index, int value);
    void ShapeIndex_push_front(ShapeIndex& shape_index, int value);

    Shape* MakeShape(int primitive_type, int* shape, int rank);
}
